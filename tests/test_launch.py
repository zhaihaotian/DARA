import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'training'))
import launch


class LaunchTest(unittest.TestCase):
    def test_all_planned_configurations_reach_hydra_with_fixed_budgets(self):
        for group in (4, 8, 16, 32):
            for method in launch.METHODS:
                rewards = ('two', 'three') if method in ('grpo', 'gdpo', 'dara') else ('two',)
                for reward in rewards:
                    args = SimpleNamespace(method=method,seed=0,group_size=group,model_size='1.5b',
                                           rewards=reward,save_freq=100,nnodes=1,ray_address=None,
                                           output=Path('/tmp/dara-test'),
                                           data_dir=launch.REPO/'data/rlla_4k',model='Qwen/Qwen2.5-1.5B-Instruct')
                    _, env, cmd = launch.configuration(args)
                    with initialize_config_dir(config_dir=str(launch.FRAMEWORK/'verl/trainer/config'),version_base=None):
                        cfg=compose(config_name='ppo_trainer',overrides=cmd[4:])
                    self.assertEqual(cfg.trainer.n_gpus_per_node,4)
                    self.assertEqual(cfg.data.train_batch_size*group,2048)
                    self.assertEqual(cfg.actor_rollout_ref.actor.ppo_mini_batch_size*group,512)
                    self.assertEqual(cfg.actor_rollout_ref.actor.ppo_micro_batch_size*group,256)
                    self.assertEqual(cfg.trainer.total_training_steps,100)
                    self.assertEqual(cfg.trainer.save_freq,100)
                    self.assertEqual(cfg.actor_rollout_ref.actor.entropy_coeff,.001)
                    self.assertEqual(env['WITHLENGTH'],'1' if reward=='three' else '0')
                    self.assertEqual(list(cfg.algorithm.reward_channels),['correctness','format']+(['length'] if reward=='three' else []))

    def test_relative_model_path_and_dry_run_are_safe_to_execute_from_elsewhere(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'model').mkdir()
            proc=subprocess.run([sys.executable,str(launch.REPO/'training/launch.py'),
                                 '--method','dara','--seed','0','--model','./model',
                                 '--model-size','3b','--output','./run','--dry-run'],
                                cwd=root,capture_output=True,text=True,check=True)
            record=json.loads(proc.stdout)
            self.assertEqual(record['settings']['actor_rollout_ref.model.path'],str(root/'model'))
            self.assertEqual(record['environment']['PYTORCH_CUDA_ALLOC_CONF'],'expandable_segments:True')
            self.assertFalse((root/'run').exists())

    def test_manifest_contains_exact_requested_work(self):
        group=json.loads((launch.REPO/'configs/group_ablation.json').read_text())['runs']
        three=json.loads((launch.REPO/'configs/three_rewards.json').read_text())['runs']
        self.assertEqual(len({r['id'] for r in group}),60)
        self.assertEqual(sum(r['status']=='planned' for r in group),45)
        self.assertEqual(len({r['id'] for r in three}),30)
        self.assertEqual({r['seed'] for r in group+three},{0,1,2,4,5})
        self.assertEqual({r['method'] for r in group+three},{'grpo','gdpo','dara'})
        self.assertTrue(all(r['gpus']==4 and r['steps']==100 for r in group+three))

    def test_periodic_checkpoint_option_preserves_training_configuration(self):
        command = [sys.executable, str(launch.REPO/'training/launch.py'),
                   '--method', 'dvao', '--seed', '0', '--model', 'Qwen/Qwen2.5-1.5B-Instruct',
                   '--model-size', '1.5b', '--output', '/tmp/dara-dvao-save10', '--dry-run']
        default = json.loads(subprocess.check_output(command, text=True))
        periodic = json.loads(subprocess.check_output(command + ['--save-freq', '10'], text=True))
        self.assertEqual(periodic['settings']['trainer.save_freq'], 10)
        default['settings']['trainer.save_freq'] = 10
        self.assertEqual(periodic['settings'], default['settings'])

    def test_two_nodes_preserve_four_gpu_training_budget(self):
        command = [sys.executable, str(launch.REPO/'training/launch.py'),
                   '--method', 'dvao', '--seed', '1', '--model', 'Qwen/Qwen2.5-1.5B-Instruct',
                   '--model-size', '1.5b', '--output', '/tmp/dara-pair', '--save-freq', '10', '--dry-run']
        single = json.loads(subprocess.check_output(command, text=True))
        pair = json.loads(subprocess.check_output(command + ['--nnodes', '2', '--ray-address', '10.1.2.3:23456'], text=True))
        self.assertEqual(pair['settings']['trainer.nnodes'], 2)
        self.assertEqual(pair['settings']['trainer.n_gpus_per_node'], 2)
        single['settings']['trainer.nnodes'] = 2
        single['settings']['trainer.n_gpus_per_node'] = 2
        self.assertEqual(single['settings'], pair['settings'])
        self.assertEqual(pair['environment']['RAY_ADDRESS'], '10.1.2.3:23456')
        self.assertEqual(pair['environment']['SEED'], '1')
        single['environment'].pop('RAY_TMPDIR')
        pair['environment'].pop('RAY_ADDRESS')
        self.assertEqual(single['environment'], pair['environment'])


if __name__=='__main__': unittest.main()
