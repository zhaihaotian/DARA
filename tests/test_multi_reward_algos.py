"""CPU checks for the migrated algorithms and the real legacy PPO actor route.

Run in the rdgdpo environment with PYTHONPATH=vendor/verl. Only the toy
actor's forward pass and its CUDA transfer are replaced; DataProto selection,
dynamic microbatching, advantages, clipped loss, and backward are real code.
"""

import math
from types import MethodType
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.multi_reward_algos import METHODS
from verl.trainer.ppo.ray_trainer import apply_kl_penalty, compute_advantage
from verl.utils.seqlen_balancing import rearrange_micro_batches
from verl.workers.actor.dp_actor import DataParallelPPOActor


def make_batch(scores=None, lengths=None):
    if scores is None:
        scores = np.array([[-3, 0], [-1, 1], [1, 0], [3, 1],
                           [0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float32)
    else:
        scores = np.asarray(scores, dtype=np.float32)
    n = len(scores)
    lengths = np.array(lengths if lengths is not None else [1, 2, 4, 3, 5, 2, 1, 4])
    mask = torch.arange(5).unsqueeze(0) < torch.tensor(lengths).unsqueeze(-1)
    reward = torch.zeros(n, 5, 2)
    reward[torch.arange(n), torch.tensor(lengths) - 1] = torch.tensor(scores)
    attention = torch.cat([torch.ones(n, 2, dtype=torch.bool), mask], dim=-1)
    batch = DataProto.from_dict(tensors={
        'responses': torch.zeros(n, 5, dtype=torch.long),
        'input_ids': torch.arange(n).unsqueeze(-1).expand(n, 7).clone(),
        'position_ids': torch.arange(7).unsqueeze(0).expand(n, 7).clone(),
        'attention_mask': attention,
        'old_log_probs': torch.zeros(n, 5),
        'ref_log_prob': torch.full((n, 5), -0.2),
        'token_level_scores_correctness': reward[:, :, 0],
        'token_level_scores_format': reward[:, :, 1],
        'token_level_scores': reward.sum(-1),
        'token_level_rewards': reward.sum(-1),
    }, non_tensors={'uid': np.array([f'g{i // 4}' for i in range(n)], dtype=object)},
        meta_info={'temperature': 1.0})
    return batch, scores, mask.numpy()


def numpy_reference(method, scores, mask, uids, k=1.0):
    scores = np.asarray(scores, dtype=np.float64)
    groups = [np.flatnonzero(uids == uid) for uid in dict.fromkeys(uids)]
    z = np.zeros_like(scores)
    for rows in groups:
        z[rows] = (scores[rows] - scores[rows].mean(0)) / (scores[rows].std(0, ddof=1) + 1e-6)
    weights = np.ones(len(scores))
    keep = np.ones(len(scores), dtype=bool)
    if method == 'dvao':
        result = np.zeros(len(scores))
        for rows in groups:
            values = scores[rows]
            denominator = 0.5 * values.std(0, ddof=0).sum()
            if denominator > 1e-8:
                result[rows] = 0.5 * (values - values.mean(0)).sum(-1) / denominator
        return result[:, None] * mask, weights
    if method == 'gdpo_saw':
        shifted = scores - np.array([-3, 0]) + 1e-8
        cv = shifted.std(0, ddof=1) / np.maximum(shifted.mean(0), 1e-8)
        channel_weights = 2 * cv / cv.sum() if cv.sum() > 1e-8 else np.ones(2)
        values = (z * channel_weights).sum(-1)
    elif method == 'rvpo':
        values = -(np.logaddexp(-k * z[:, 0], -k * z[:, 1]) - math.log(2)) / k
    elif method == 'gd2po_hard':
        keep = ~((z > 1e-8).any(-1) & (z < -1e-8).any(-1))
        for rows in groups:
            weights[rows] = keep[rows].mean()
        values = z.sum(-1)
    else:
        raise ValueError(method)
    token_values = np.broadcast_to(values[:, None], mask.shape)
    keep_mask = mask & keep[:, None]
    valid = token_values[keep_mask]
    out = np.zeros_like(token_values)
    if valid.size > 1:
        out[keep_mask] = (valid - valid.mean()) / np.sqrt(valid.var(ddof=1) + 1e-8)
    return out, weights


class MigratedAdvantagesTest(unittest.TestCase):
    def test_all_estimators_match_independent_numpy_token_reference(self):
        for method in METHODS:
            with self.subTest(method=method):
                batch, scores, mask = make_batch()
                # Noncontiguous prompt groups exercise UID grouping after balancing.
                order = np.array([4, 0, 5, 1, 6, 2, 7, 3])
                batch.reorder(torch.tensor(order))
                output = compute_advantage(batch, method, algorithm_config={'rvpo': {'k': 1.0}})
                expected, weights = numpy_reference(method, scores[order], mask[order], output.non_tensor_batch['uid'])
                np.testing.assert_allclose(output.batch['advantages'].numpy(), expected, atol=2e-6)
                torch.testing.assert_close(output.batch['returns'], output.batch['advantages'], rtol=0, atol=0)
                if method == 'gd2po_hard':
                    np.testing.assert_array_equal(output.batch['gd2po_query_weight'].numpy(), weights)
                else:
                    self.assertNotIn('gd2po_query_weight', output.batch)

    def test_dvao_population_std_and_no_final_whitening(self):
        batch, _, mask = make_batch()
        actual = compute_advantage(batch, 'dvao').batch['advantages'].numpy()
        self.assertGreater(abs(actual[mask].mean()), 0.01)
        self.assertGreater(abs(actual[mask].std(ddof=1) - 1), 0.01)

    def test_saw_weights_sum_to_two_and_use_shifted_batch_cv(self):
        batch, scores, _ = make_batch()
        out = compute_advantage(batch, 'gdpo_saw')
        shifted = scores.astype(np.float64) - [-3, 0] + 1e-8
        cvs = shifted.std(0, ddof=1) / shifted.mean(0)
        weights = 2 * cvs / cvs.sum()
        metrics = out.meta_info['multi_reward_metrics']
        self.assertAlmostEqual(metrics['gdpo_saw/weight_sum'], 2, places=6)
        for i, name in enumerate(('correctness', 'format')):
            self.assertAlmostEqual(metrics[f'gdpo_saw/weight_{name}'], weights[i], places=6)

    def test_hard_neutral_channels_keep_responses_and_conflicts_stay_zero(self):
        batch, _, _ = make_batch()
        out = compute_advantage(batch, 'gd2po_hard')
        torch.testing.assert_close(out.batch['gd2po_query_weight'], torch.tensor([.5] * 4 + [1.] * 4))
        self.assertEqual(out.batch['advantages'][[1, 2]].abs().sum().item(), 0)
        self.assertEqual(out.meta_info['multi_reward_metrics']['gd2po_hard/conflict_ratio'], .25)

    def test_all_conflicts_and_all_constant_channels_are_finite(self):
        scores = [[-3, 1], [-1, 1], [1, 0], [3, 0]] * 2
        out = compute_advantage(make_batch(scores)[0], 'gd2po_hard')
        self.assertEqual(out.batch['advantages'].abs().sum().item(), 0)
        self.assertEqual(out.batch['gd2po_query_weight'].sum().item(), 0)
        for method in METHODS:
            with self.subTest(method=method):
                out = compute_advantage(make_batch([[1, 0]] * 8)[0], method)
                self.assertTrue(torch.isfinite(out.batch['advantages']).all())
                self.assertEqual(out.batch['advantages'].abs().sum().item(), 0)

    def test_rvpo_k_config_reaches_trainer_and_constant_channel_is_applicable(self):
        batch, scores, mask = make_batch()
        out = compute_advantage(batch, 'rvpo', algorithm_config=OmegaConf.create({'rvpo': {'k': 2.0}}))
        expected, _ = numpy_reference('rvpo', scores, mask, out.non_tensor_batch['uid'], k=2)
        np.testing.assert_allclose(out.batch['advantages'].numpy(), expected, atol=2e-6)
        self.assertEqual(out.meta_info['multi_reward_metrics']['rvpo/k'], 2)
        self.assertGreater(out.meta_info['multi_reward_metrics']['rvpo/variance_penalty_mean'], 0)

    def test_legacy_kl_reward_bypass_is_preserved(self):
        for method in ('gdpo', 'dara', *METHODS):
            with self.subTest(method=method):
                original = make_batch()[0]
                penalized = make_batch()[0]
                penalized.batch['old_log_probs'] = torch.arange(40).reshape(8, 5).float() / 20
                penalized, _ = apply_kl_penalty(penalized, core_algos.FixedKLController(.1))
                self.assertFalse(torch.equal(penalized.batch['token_level_rewards'], original.batch['token_level_rewards']))
                a = compute_advantage(original, method).batch['advantages']
                b = compute_advantage(penalized, method).batch['advantages']
                torch.testing.assert_close(a, b, rtol=0, atol=0)


class LegacyActorRouteTest(unittest.TestCase):
    def test_clipped_query_weights_preserve_denominator_and_gradient(self):
        advantage = torch.tensor([[2., -1., 0.], [-2., 3., 1.]])
        mask = torch.tensor([[1., 1., 0.], [1., 1., 1.]])
        weight = torch.tensor([.25, .75])
        log_prob = torch.tensor([[.4, -.4, .1], [.3, -.1, -.4]], requires_grad=True)
        old = torch.zeros_like(log_prob)
        got, clip, kl = core_algos.compute_policy_loss(old, log_prob, advantage, mask, .2, weight)
        ratio = log_prob.exp()
        expected = (torch.maximum(-advantage * ratio, -advantage * ratio.clamp(.8, 1.2))
                    * weight[:, None] * mask).sum() / mask.sum()
        torch.testing.assert_close(got, expected)
        torch.testing.assert_close(torch.autograd.grad(got, log_prob, retain_graph=True)[0],
                                   torch.autograd.grad(expected, log_prob)[0])
        plain, plain_clip, plain_kl = core_algos.compute_policy_loss(old, log_prob, advantage, mask, .2)
        self.assertNotEqual(plain.item(), got.item())
        torch.testing.assert_close(clip, plain_clip, rtol=0, atol=0)
        torch.testing.assert_close(kl, plain_kl, rtol=0, atol=0)

    def test_actor_dataproto_weight_routing_and_legacy_dynamic_accumulation(self):
        for method in ('gdpo', 'gd2po_hard'):
            with self.subTest(method=method):
                batch = compute_advantage(make_batch()[0], method)
                module = torch.nn.Linear(1, 1, bias=False)
                with torch.no_grad():
                    module.weight.fill_(0.13)
                actor = object.__new__(DataParallelPPOActor)
                actor.config = OmegaConf.create(dict(
                    ppo_mini_batch_size=8, ppo_micro_batch_size=2,
                    use_dynamic_bsz=True, ppo_max_token_len_per_gpu=14,
                    use_kl_loss=False, clip_ratio=.2, entropy_coeff=.001,
                ))
                actor.ulysses_sequence_parallel_size = 1
                actor.actor_module = module
                actor.actor_optimizer = torch.optim.SGD(module.parameters(), lr=.1)
                seen_rows, seen_weights, gradients = [], [], []

                def forward(self, micro_batch, temperature):
                    features = (micro_batch['input_ids'][:, :1].float() - 3.5) / 4
                    log_prob = self.actor_module.weight.reshape(()) * features.expand(-1, 5)
                    seen_rows.extend(micro_batch['input_ids'][:, 0].tolist())
                    if 'gd2po_query_weight' in micro_batch:
                        seen_weights.extend(micro_batch['gd2po_query_weight'].tolist())
                    return log_prob * .07 + 1, log_prob

                def optimizer_step(self):
                    gradient = self.actor_module.weight.grad.detach().clone()
                    gradients.append(gradient)
                    self.actor_optimizer.step()
                    return gradient.norm()

                actor._forward_micro_batch = MethodType(forward, actor)
                actor._optimizer_step = MethodType(optimizer_step, actor)
                expected_parameter = torch.tensor(.13, requires_grad=True)
                pieces, _ = rearrange_micro_batches(batch.batch, max_token_len=14)
                expected_loss = 0
                for piece in pieces:
                    features = (piece['input_ids'][:, :1].float() - 3.5) / 4
                    logp = expected_parameter * features.expand(-1, 5)
                    ratio = logp.exp()
                    mask = piece['attention_mask'][:, -5:]
                    values = torch.maximum(-piece['advantages'] * ratio,
                                           -piece['advantages'] * ratio.clamp(.8, 1.2))
                    if method == 'gd2po_hard':
                        values = values * piece['gd2po_query_weight'].unsqueeze(-1)
                    policy = (values * mask).sum() / mask.sum()
                    entropy = ((logp * .07 + 1) * mask).sum() / mask.sum()
                    expected_loss = expected_loss + (policy - .001 * entropy) / 4
                expected_gradient = torch.autograd.grad(expected_loss, expected_parameter)[0]
                with patch.object(TensorDict, 'cuda', lambda self, *args, **kwargs: self):
                    metrics = actor.update_policy(batch)
                self.assertEqual(sorted(seen_rows), list(range(8)))
                self.assertEqual(actor.gradient_accumulation, 4)
                torch.testing.assert_close(gradients[0].reshape(()), expected_gradient)
                self.assertEqual(len(metrics['actor/pg_loss']), len(pieces))
                if method == 'gd2po_hard':
                    expected_weights = batch.batch['gd2po_query_weight'][seen_rows].tolist()
                    self.assertEqual(seen_weights, expected_weights)
                else:
                    self.assertEqual(seen_weights, [])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
