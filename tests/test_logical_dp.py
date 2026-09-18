import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from verl.utils.logical_dp import RolloutStreams
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'training'))
from check_logical_dp import fixed_batch


class LogicalDPTests(unittest.TestCase):
    def test_trainer_preserves_four_rank_data_membership_on_two_devices(self):
        results = []
        for physical in (4, 2):
            trainer = object.__new__(RayPPOTrainer)
            trainer.config = OmegaConf.create(dict(trainer={'logical_world_size': 4} if physical==2 else {}))
            trainer.actor_rollout_wg = SimpleNamespace(world_size=physical)
            batch = fixed_batch()
            trainer._balance_batch(batch, {})
            results.append(batch.batch['input_ids'][:, 0])
        torch.testing.assert_close(*results, rtol=0, atol=0)

    def test_logical_rollouts_keep_independent_rng_streams_and_request_counters(self):
        fake_cuda = torch.Generator().manual_seed(321)
        engine = SimpleNamespace(request_counter=SimpleNamespace(counter=0),
                                 llm_engine=SimpleNamespace(seq_counter=SimpleNamespace(counter=0)))
        references = [torch.Generator().manual_seed(1000+i) for i in (2, 3)]
        torch.manual_seed(432)
        random.seed(432)
        np.random.seed(432)
        original_cpu = torch.get_rng_state()
        original_cuda = fake_cuda.get_state()
        with patch.object(torch.cuda, 'get_rng_state', fake_cuda.get_state), \
             patch.object(torch.cuda, 'set_rng_state', fake_cuda.set_state), \
             patch.object(torch.cuda, 'manual_seed', fake_cuda.manual_seed):
            streams = RolloutStreams(engine, 2, 2)
            counts = [0, 0]
            for step in range(3):
                for index in (0, 1):
                    with streams.use(index):
                        self.assertEqual(engine.request_counter.counter, counts[index])
                        self.assertEqual(engine.llm_engine.seq_counter.counter, 4*counts[index])
                        n = 3+index+step
                        actual = torch.rand(n, generator=fake_cuda)
                        expected = torch.rand(n, generator=references[index])
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        torch.rand(2)
                        np.random.rand(2)
                        random.random()
                        engine.request_counter.counter += n
                        engine.llm_engine.seq_counter.counter += 4*n
                        counts[index] += n
            torch.testing.assert_close(torch.get_rng_state(), original_cpu, rtol=0, atol=0)
            torch.testing.assert_close(fake_cuda.get_state(), original_cuda, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
