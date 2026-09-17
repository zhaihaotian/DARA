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
                                           rewards=reward,output=Path('/tmp/dara-test'),
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


if __name__=='__main__': unittest.main()
