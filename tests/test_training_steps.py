"""Exercise the real training loop without model execution."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class TrainingStepsTest(unittest.TestCase):
    def run_loop(self, steps, save_freq=100):
        trainer = object.__new__(RayPPOTrainer)
        trainer.config = OmegaConf.create({
            'trainer': {'project_name': 'test', 'experiment_name': 'steps', 'logger': ['console'],
                        'total_epochs': 1, 'test_freq': 10, 'save_freq': save_freq, 'critic_warmup': 0},
            'actor_rollout_ref': {'rollout': {'n': 1}, 'actor': {'use_kl_loss': True}},
            'algorithm': {'adv_estimator': 'gdpo', 'gamma': 1.0, 'lam': 1.0},
        })
        trainer.total_training_steps = steps
        trainer.train_dataloader = [dict(
            input_ids=torch.ones(1, 1, dtype=torch.long),
            attention_mask=torch.ones(1, 1, dtype=torch.long),
            position_ids=torch.zeros(1, 1, dtype=torch.long),
        ) for _ in range(steps + 5)]
        trainer.use_reference_policy = trainer.use_critic = trainer.use_rm = False
        trainer._balance_batch = Mock()
        trainer.reward_fn = lambda batch, step: (torch.zeros(1, 2),) * 4
        validations, saves, updates = [], [], []
        trainer.val_reward_fn = True
        trainer._validate = lambda: validations.append(trainer.global_steps) or {'val/test': 1.0}
        trainer._save_checkpoint = lambda: saves.append(trainer.global_steps)

        def generate(batch):
            return DataProto.from_dict({
                'prompts': torch.ones(1, 1, dtype=torch.long),
                'responses': torch.ones(1, 1, dtype=torch.long),
                'attention_mask': torch.ones(1, 2, dtype=torch.long),
            })

        def update(batch):
            updates.append(trainer.global_steps)
            return SimpleNamespace(meta_info={'metrics': {'actor/pg_loss': [0.1]}})

        trainer.actor_rollout_wg = SimpleNamespace(generate_sequences=generate, update_actor=update)
        with patch('verl.utils.tracking.Tracking') as tracking, \
                patch('verl.trainer.ppo.ray_trainer.compute_advantage', side_effect=lambda batch, **kwargs: batch), \
                patch('verl.trainer.ppo.ray_trainer.compute_data_metrics', return_value={}), \
                patch('verl.trainer.ppo.ray_trainer.compute_timing_metrics', return_value={}):
            trainer.fit()
        train_steps = [call.kwargs['step'] for call in tracking.return_value.log.call_args_list
                       if 'actor/pg_loss' in call.kwargs['data']]
        self.assertEqual(updates, list(range(1, steps + 1)))
        self.assertEqual(train_steps, updates)
        return validations, saves

    def test_one_hundred_steps_and_single_final_checkpoint(self):
        validations, saves = self.run_loop(100)
        self.assertEqual(validations, list(range(0, 101, 10)))
        self.assertEqual(saves, [100])

    def test_final_step_between_periodic_evaluations(self):
        validations, saves = self.run_loop(23)
        self.assertEqual(validations, [0, 10, 20, 23])
        self.assertEqual(saves, [23])

    def test_save_every_ten_steps(self):
        validations, saves = self.run_loop(100, save_freq=10)
        self.assertEqual(validations, list(range(0, 101, 10)))
        self.assertEqual(saves, list(range(10, 101, 10)))


if __name__ == '__main__':
    unittest.main()
