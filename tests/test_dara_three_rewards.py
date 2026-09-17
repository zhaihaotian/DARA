import contextlib
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch

from test_multi_reward_algos import make_batch
from verl.trainer.ppo.core_algos import compute_dara_combined_advantage
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.utils.reward_score import rlla


class ThreeRewardsTest(unittest.TestCase):
    def test_density_scales_both_signs_and_logs_each_channel(self):
        channels = [torch.tensor([[1.], [-1.], [1.], [-1.]]),
                    torch.tensor([[1.], [-1.], [0.], [0.]]), torch.zeros(4, 1)]
        result, metrics = compute_dara_combined_advantage(channels, np.array(['a','a','b','b']),
                                                          ['correctness', 'format', 'length'])
        expected = channels[0] + 2**0.5 * channels[1]
        torch.testing.assert_close(result, expected)
        self.assertEqual(metrics['dara/pi_format'], 0.5)
        self.assertEqual(metrics['dara/pi_length'], 0)
        self.assertEqual(metrics['dara/w_length'], 5)

    def test_third_channel_changes_every_core_estimator(self):
        for method in ('grpo', 'gdpo', 'dara'):
            data = make_batch()[0]
            n, width = data.batch['responses'].shape
            length = torch.zeros(n, width)
            length[:, 0] = torch.tensor([0., 1., 0., 0., 1., 0., 0., 1.])
            data.batch['token_level_scores_length'] = length
            original = compute_advantage(data, method).batch['advantages'].clone()
            data.batch['token_level_rewards'] += length
            updated = compute_advantage(data, method, algorithm_config={
                'reward_channels': ['correctness', 'format', 'length']}).batch['advantages']
            self.assertFalse(torch.allclose(original, updated), method)
            self.assertTrue(torch.isfinite(updated).all())

    def test_constant_length_has_no_gdpo_or_dara_signal(self):
        for method in ('gdpo', 'dara'):
            data = make_batch()[0]
            data.batch['token_level_scores_length'] = torch.zeros_like(data.batch['token_level_scores_format'])
            two = compute_advantage(data, method).batch['advantages'].clone()
            three = compute_advantage(data, method, algorithm_config={
                'reward_channels': ['correctness', 'format', 'length']}).batch['advantages']
            torch.testing.assert_close(two, three, rtol=0, atol=0)

    def test_real_reward_scorer_adds_length_without_changing_other_channels(self):
        text = '<think>' + 'word '*512 + '</think>\n<tool_call>\n[]\n</tool_call>'
        with patch.dict(os.environ, {'EXPERIMENT_NAME':'qwen-test', 'SCHEDULELENGTH':'0',
                                     'WITHLENGTH':'0'}), contextlib.redirect_stdout(io.StringIO()):
            base = rlla.compute_score(text, text)
            with patch.dict(os.environ, {'WITHLENGTH':'1'}):
                three = rlla.compute_score(text, text)
        self.assertEqual(base[1:3], three[1:3])
        self.assertEqual(three[3], 1)
        self.assertEqual(three[0], base[0]+1)


if __name__ == '__main__':
    unittest.main()
