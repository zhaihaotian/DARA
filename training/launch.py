"""Launch the Haotian Tool Calling comparison from one checkout."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
FRAMEWORK = REPO / 'vendor' / 'verl'
METHODS = ('grpo', 'gdpo', 'dara', 'dvao', 'gdpo_saw', 'gd2po_hard', 'rvpo')


def configuration(args):
    group = args.group_size
    gpus = getattr(args, 'gpus', 4)
    if args.nnodes == 2 and gpus != 4:
        raise ValueError('Paired allocations use four GPUs in total.')
    if args.nnodes == 2 and not args.ray_address:
        raise ValueError('Two-node training requires --ray-address for the four-GPU Ray cluster.')
    estimator = args.method
    if args.rewards == 'three' and args.method not in ('grpo', 'gdpo', 'dara'):
        raise ValueError('The three-reward experiment is defined for GRPO, GDPO and DARA.')
    output = args.output.resolve()
    data = args.data_dir.resolve()
    model = str(Path(args.model).resolve()) if Path(args.model).exists() else args.model
    settings = {
        'algorithm.adv_estimator': estimator,
        '+algorithm.reward_channels': ['correctness', 'format'] + (['length'] if args.rewards == 'three' else []),
        'data.train_files': str(data/'train.parquet'),
        'data.val_files': str(data/'test.parquet'),
        'data.train_batch_size': 2048//group,
        'data.val_batch_size': 128,
        'data.max_prompt_length': 2048,
        'data.max_response_length': 1024,
        'actor_rollout_ref.model.path': model,
        'actor_rollout_ref.actor.optim.lr': 1e-6,
        'actor_rollout_ref.model.use_remove_padding': True,
        'actor_rollout_ref.actor.ppo_mini_batch_size': 512//group,
        'actor_rollout_ref.actor.ppo_micro_batch_size': 256//group,
        'actor_rollout_ref.actor.use_dynamic_bsz': True,
        'actor_rollout_ref.actor.ppo_max_token_len_per_gpu': 16384,
        'actor_rollout_ref.actor.use_kl_loss': False,
        'actor_rollout_ref.actor.kl_loss_coef': 0.001,
        'actor_rollout_ref.actor.kl_loss_type': 'low_var_kl',
        'actor_rollout_ref.model.enable_gradient_checkpointing': True,
        'actor_rollout_ref.actor.fsdp_config.param_offload': False,
        'actor_rollout_ref.actor.fsdp_config.grad_offload': False,
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload': False,
        'actor_rollout_ref.rollout.tensor_model_parallel_size': 1,
        'actor_rollout_ref.rollout.name': 'vllm',
        'actor_rollout_ref.rollout.gpu_memory_utilization': 0.6,
        'actor_rollout_ref.rollout.n': group,
        'actor_rollout_ref.ref.fsdp_config.param_offload': True,
        'algorithm.kl_ctrl.kl_coef': 0.001,
        'trainer.critic_warmup': 0,
        'trainer.logger': ['console', 'jsonl'],
        'trainer.project_name': 'Var_inspect',
        'trainer.experiment_name': f'qwen-{args.model_size}-{args.method}-s{args.seed}-G{group}-{args.rewards}-100',
        'trainer.n_gpus_per_node': gpus // args.nnodes,
        'trainer.nnodes': args.nnodes,
        'trainer.save_freq': args.save_freq,
        'trainer.test_freq': 10,
        'trainer.default_local_dir': str(output),
        'trainer.total_training_steps': 100,
        'trainer.total_epochs': 15,
    }
    method_settings = {
        'dara': {'+algorithm.dara.w_max': 5.0},
        'dvao': {'+algorithm.dvao.base_weights': [0.5, 0.5],
                 '+algorithm.dvao.denominator_epsilon': 1e-8},
        'gdpo_saw': {'+algorithm.gdpo_saw.theoretical_minima': [-3.0, 0.0],
                     '+algorithm.gdpo_saw.cv_epsilon': 1e-8},
        'gd2po_hard': {'+algorithm.gd2po_hard.sign_epsilon': 1e-8},
        'rvpo': {'+algorithm.rvpo.k': 1.0},
    }
    settings.update(method_settings.get(args.method, {}))
    if gpus == 2:
        settings['+actor_rollout_ref.logical_world_size'] = 4
        settings['+trainer.logical_world_size'] = 4
    environment = dict(SEED=str(args.seed),
                       EXPERIMENT_NAME=settings['trainer.experiment_name'],
                       DATA_DIR=str(data), BASE_MODEL=model, CKPT_DIR=str(output),
                       PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                       TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
                       VLLM_ATTENTION_BACKEND='XFORMERS',
                       N_GPUS=str(gpus), ROLLOUT_TP_SIZE='1',
                       RAY_USAGE_STATS_ENABLED='0', RAY_DISABLE_DOCKER_CPU_WARNING='1')
    if args.ray_address:
        environment['RAY_ADDRESS'] = args.ray_address
    else:
        environment['RAY_TMPDIR'] = f'/tmp/d{os.getuid()}p{os.getpid()}'
    for name in ('WITHLENGTH', 'REFINEDREWARD', 'COARSEREWARD', 'STRICTMATCH',
                 'CORRECTMAX1', 'MAX1STEP30MAX3', 'SCHEDULEREWARD', 'SCHEDULELENGTH',
                 'INTERMEDIATEREWARD', 'FORMAT_GRADED'):
        environment[name] = '0'
    environment['WITHLENGTH'] = '1' if args.rewards == 'three' else '0'
    if args.model_size == '3b':
        environment['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    command = [sys.executable, '-u', '-m', 'verl.trainer.main_ppo']
    command += [f'{key}={json.dumps(value, separators=(",", ":"))}' for key, value in settings.items()]
    return settings, environment, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--seed', type=int, choices=[0, 1, 2, 3, 4, 5], required=True)
    parser.add_argument('--model', required=True, help='Qwen2.5-1.5B/3B-Instruct path or model ID')
    parser.add_argument('--model-size', choices=['1.5b', '3b'], required=True)
    parser.add_argument('--data-dir', type=Path, default=REPO/'data'/'rlla_4k')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rewards', choices=['two', 'three'], default='two')
    parser.add_argument('--group-size', type=int, choices=[4, 8, 16, 32], default=4)
    parser.add_argument('--save-freq', type=int, choices=[10, 100], default=100,
                        help='Save a Hugging Face model every N training steps')
    parser.add_argument('--nnodes', type=int, choices=[1, 2], default=1,
                        help='Split the fixed four GPUs over one or two Ray nodes')
    parser.add_argument('--gpus', type=int, choices=[2, 4], default=4,
                        help='Physical GPUs per run; two GPUs execute the original four logical data shards')
    parser.add_argument('--ray-address', help='Address of an existing Ray head')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    settings, environment, command = configuration(args)
    record = dict(method=args.method, seed=args.seed, group_size=args.group_size,
                  model_size=args.model_size, rewards=args.rewards,
                  gpus=args.gpus, responses_per_step=2048, optimizer_updates_per_step=4,
                  rollout_seed=0, settings=settings, environment=environment, command=command)
    if args.dry_run:
        print(json.dumps(record, indent=2))
        return
    for name in ('train.parquet', 'test.parquet'):
        if not (args.data_dir/name).is_file():
            raise FileNotFoundError(args.data_dir/name)
    if (args.output/'metrics.jsonl').exists():
        raise FileExistsError(f'Existing training results: {args.output}; choose a new output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'launch.json').write_text(json.dumps(record, indent=2)+'\n')
    (args.output/'command.json').write_text(json.dumps(settings, indent=2)+'\n')
    env = os.environ.copy()
    for key in ('ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'RAY_ADDRESS'):
        env.pop(key, None)
    env.update(environment)
    env['PYTHONPATH'] = str(FRAMEWORK) + (os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    result = subprocess.run(command, cwd=FRAMEWORK, env=env)
    (args.output/'exit.code').write_text(str(result.returncode)+'\n')
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
