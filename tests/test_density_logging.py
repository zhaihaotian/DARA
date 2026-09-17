"""Check channel diagnostics and their path into the per-step JSONL log."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from omegaconf import OmegaConf
import torch

from test_multi_reward_algos import make_batch
from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, apply_kl_penalty, compute_advantage
from verl.utils.torch_functional import masked_whiten


def density_batch(three=False):
    batch = make_batch()[0]
    length = torch.zeros_like(batch.batch['token_level_scores_format'])
    if three:
        length[:, 0] = torch.tensor([0., 0., 0., 0., 0., 1., 0., 1.])
    batch.batch['token_level_scores_length'] = length
    batch.batch['token_level_scores'] = batch.batch['token_level_scores'] + length
    batch.batch['token_level_rewards'] = batch.batch['token_level_scores'].clone()
    return batch


class DensityLoggingTest(unittest.TestCase):
    def test_core_methods_log_matching_densities_after_uid_reordering(self):
        for three in (False, True):
            names = ['correctness', 'format'] + (['length'] if three else [])
            for method in ('grpo', 'gdpo', 'dara'):
                with self.subTest(three=three, method=method):
                    batch = density_batch(three)
                    batch.reorder(torch.tensor([4, 0, 5, 1, 6, 2, 7, 3]))
                    output = compute_advantage(batch, method, algorithm_config={'reward_channels': names})
                    key = 'dara_metrics' if method == 'dara' else 'density_metrics'
                    metrics = output.meta_info[key]
                    expected = {}
                    for name in names:
                        pi = 1.0 if name == 'correctness' else 0.5
                        expected[f'{method}/pi_{name}'] = pi
                        expected[f'{method}/active_groups_{name}'] = pi * 2
                        expected[f'{method}/w_{name}'] = (1 / pi)**0.5 if method == 'dara' else 1.0
                    self.assertEqual(metrics, expected)

    def test_constant_and_cancelling_channels_are_diagnosed_separately(self):
        # The first group's correctness and format vary but sum to zero.
        scores = [[-1, 1], [0, 0], [-1, 1], [0, 0]] + [[0, 0]] * 4
        for method in ('grpo', 'gdpo'):
            with self.subTest(method=method):
                batch = make_batch(scores)[0]
                batch.batch['token_level_scores_length'] = torch.zeros_like(batch.batch['responses']).float()
                output = compute_advantage(batch, method, algorithm_config={
                    'reward_channels': ['correctness', 'format', 'length']})
                metrics = output.meta_info['density_metrics']
                for name in ('correctness', 'format'):
                    self.assertEqual(metrics[f'{method}/pi_{name}'], 0.5)
                    self.assertEqual(metrics[f'{method}/active_groups_{name}'], 1)
                self.assertEqual(metrics[f'{method}/pi_length'], 0)
                self.assertEqual(metrics[f'{method}/active_groups_length'], 0)
                self.assertEqual(metrics[f'{method}/w_length'], 1)
                self.assertEqual(output.batch['advantages'].abs().sum().item(), 0)

    def test_logging_preserves_advantages_returns_and_input_tensors(self):
        for three in (False, True):
            names = ['correctness', 'format'] + (['length'] if three else [])
            for method in ('grpo', 'gdpo', 'dara'):
                with self.subTest(three=three, method=method):
                    batch = density_batch(three)
                    batch.batch['old_log_probs'] = torch.arange(40).reshape(8, 5).float() / 20
                    batch, _ = apply_kl_penalty(batch, core_algos.FixedKLController(0.1))
                    original = {key: value.clone() for key, value in batch.batch.items()}
                    mask = batch.batch['attention_mask'][:, -5:]
                    index = batch.non_tensor_batch['uid']
                    if method == 'grpo':
                        expected, _ = core_algos.compute_grpo_outcome_advantage(
                            original['token_level_rewards'], mask, index)
                    else:
                        channels = [core_algos.compute_grpo_outcome_advantage(
                            original['token_level_scores_' + name], mask, index)[0] for name in names]
                        expected = channels[0]
                        for channel in channels[1:]:
                            expected = expected + (2**0.5 * channel if method == 'dara' else channel)
                        expected = masked_whiten(expected, mask) * mask
                    output = compute_advantage(batch, method, algorithm_config={'reward_channels': names})
                    for field in ('advantages', 'returns'):
                        torch.testing.assert_close(output.batch[field], expected, rtol=0, atol=0)
                    for key, value in original.items():
                        torch.testing.assert_close(output.batch[key], value, rtol=0, atol=0)

    def test_training_loop_writes_density_metrics_on_every_step(self):
        for three in (False, True):
            names = ['correctness', 'format'] + (['length'] if three else [])
            for method in ('grpo', 'gdpo', 'dara'):
                with self.subTest(three=three, method=method), tempfile.TemporaryDirectory() as tmp:
                    sample = density_batch(three).batch
                    trainer = object.__new__(RayPPOTrainer)
                    trainer.config = OmegaConf.create({
                        'trainer': {'project_name': 'test', 'experiment_name': method, 'logger': ['jsonl'],
                                    'default_local_dir': tmp, 'total_epochs': 1, 'test_freq': 0,
                                    'save_freq': 0, 'critic_warmup': 0},
                        'actor_rollout_ref': {'rollout': {'n': 4}, 'actor': {'use_kl_loss': False}},
                        'algorithm': {'adv_estimator': method, 'gamma': 1.0, 'lam': 1.0, 'kl_penalty': 'kl',
                                      'reward_channels': names},
                    })
                    trainer.total_training_steps = 2
                    trainer.train_dataloader = [dict(
                        input_ids=torch.ones(2, 2, dtype=torch.long),
                        attention_mask=torch.ones(2, 2, dtype=torch.long),
                        position_ids=torch.zeros(2, 2, dtype=torch.long),
                    ) for _ in range(2)]
                    trainer.use_reference_policy = trainer.use_critic = trainer.use_rm = False
                    trainer.val_reward_fn = None
                    trainer.kl_ctrl = core_algos.FixedKLController(0.1)
                    trainer._balance_batch = Mock()
                    trainer.reward_fn = lambda batch, step: tuple(sample[key] for key in (
                        'token_level_scores', 'token_level_scores_format',
                        'token_level_scores_correctness', 'token_level_scores_length'))

                    def generate(batch):
                        return DataProto.from_dict({
                            'prompts': sample['input_ids'][:, :2],
                            **{key: sample[key] for key in ('responses', 'attention_mask',
                                                           'old_log_probs', 'ref_log_prob')},
                        })

                    trainer.actor_rollout_wg = SimpleNamespace(
                        generate_sequences=generate,
                        update_actor=lambda batch: SimpleNamespace(meta_info={'metrics': {'actor/pg_loss': [0.1]}}))
                    with contextlib.redirect_stdout(io.StringIO()):
                        trainer.fit()
                    records = [json.loads(line) for line in (Path(tmp)/'metrics.jsonl').read_text().splitlines()]
                    self.assertEqual([row['step'] for row in records], [1, 2])
                    for row in records:
                        for name in names:
                            pi = 1.0 if name == 'correctness' else 0.5
                            self.assertEqual(row[f'{method}/pi_{name}'], pi)
                            self.assertEqual(row[f'{method}/active_groups_{name}'], pi * 2)
                            self.assertEqual(row[f'{method}/w_{name}'], (1/pi)**0.5 if method == 'dara' else 1)
                        self.assertIn('critic/format_score/mean', row)


if __name__ == '__main__':
    unittest.main()
