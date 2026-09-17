import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('launch_pair', ROOT/'training/launch_pair.py')
pair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pair)


class PairLaunchTest(unittest.TestCase):
    def test_actual_slurm_resource_fields_are_parsed(self):
        text = ('JobId=123 JobState=RUNNING NumNodes=1 NodeList=aga21 '
                'EndTime=2050-01-01T12:00:00 '
                'AllocTRES=cpu=8,mem=96G,node=1,gres/gpu=2,gres/gpu:a100=2 ')
        with patch.object(pair.subprocess, 'check_output', return_value=text):
            job = pair.allocation('123')
        self.assertEqual((job['id'], job['node'], job['gpu_type']), ('123', 'aga21', 'a100'))
        command = pair.step_command(job, Path('/tmp/pair.json'), 'head')
        self.assertIn('--jobid=123', command)
        self.assertIn('--gres=gpu:a100:2', command)
        self.assertIn('-c8', command)
        self.assertIn('--mem=96G', command)

    def test_four_gpu_allocation_is_not_consumed_as_a_two_gpu_pair(self):
        text = ('JobId=123 JobState=RUNNING NumNodes=1 NodeList=aga21 '
                'EndTime=2050-01-01T12:00:00 '
                'AllocTRES=cpu=16,mem=192G,node=1,gres/gpu=4,gres/gpu:a100=4 ')
        with patch.object(pair.subprocess, 'check_output', return_value=text):
            with self.assertRaisesRegex(ValueError, 'exactly two GPUs'):
                pair.allocation('123')

    def test_h100_worker_step_uses_the_second_allocation(self):
        command = pair.step_command(dict(id='456', gpu_type='h100'), Path('/tmp/pair.json'), 'worker')
        self.assertIn('--jobid=456', command)
        self.assertIn('--gres=gpu:h100:2', command)
        self.assertEqual(command[command.index('--role')+1], 'worker')
        self.assertEqual(command[command.index('--allocation')+1], '456')


if __name__ == '__main__':
    unittest.main()
