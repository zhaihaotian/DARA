"""Launch one DARA math run (or a baseline). Prints the command unless --execute."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# Paper method name -> (verl estimator, DARA calibration side, static channel weights or None).
METHODS = {
    "grpo": ("grpo", None, None),
    "gdpo": ("gdpo", None, None),
    "dvao": ("dvao", None, None),
    "gd2po_hard": ("gd2po_hard", None, None),
    "dara_asym": ("rdgdpo", "positive", None),
    "dara_sym": ("rdgdpo", "symmetric", None),
    "static_asym": ("rdgdpo", "positive", "static"),
    "static_sym": ("rdgdpo", "symmetric", "static"),
}
MODELS = {
    "r1-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "qwen3-instruct": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3-thinking": "Qwen/Qwen3-4B-Thinking-2507",
    "r1-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--group-size", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--prompt-batch", type=int, default=512)
    p.add_argument("--static-length-weight", type=float, default=5.,
                   help="static_asym / static_sym: fixed length weight (correctness weight is 1)")
    p.add_argument("--w-max", type=float, default=5.)
    p.add_argument("--profile", type=Path, default=ROOT/"configs/machine.example.json")
    p.add_argument("--resume", type=Path, help="Full checkpoint directory; restores optimizer and data state")
    p.add_argument("--execute", action="store_true")
    return p


def build(args, profile):
    if profile.get("nodes", 1) != 1:
        raise ValueError("This launcher targets one node; multi-node Ray orchestration must be configured separately")
    estimator, calibration, static = METHODS[args.method]
    g = args.group_size if args.group_size is not None else 16
    steps = args.steps if args.steps is not None else 200
    if (g < 2 or steps < 1 or args.prompt_batch < 1 or args.seed < 0
            or not math.isfinite(args.w_max) or args.w_max < 1
            or not math.isfinite(args.static_length_weight) or args.static_length_weight < 1):
        raise ValueError("Need G>=2, steps/B>0, seed>=0 and finite w_max, static weight >= 1")
    framework = Path(profile["framework"]).expanduser().resolve()
    data_root = Path(profile.get("data_root", ROOT/"data")).expanduser().resolve()
    data = data_root/"deepscaler"
    model = profile.get("models", {}).get(args.model, MODELS[args.model])
    name = (f"math_{args.model}_{args.method}_g{g}_b{args.prompt_batch}_t{steps}_s{args.seed}"
            f"_w{args.w_max:g}" + (f"_c{args.static_length_weight:g}" if static else ""))
    run_dir = Path(profile["output_root"]).expanduser().resolve()/name
    ppo_batch = min(64, args.prompt_batch)
    if args.prompt_batch % ppo_batch:
        raise ValueError("prompt-batch must be divisible by the PPO prompt mini-batch")
    overrides = {
        "algorithm.adv_estimator": estimator,
        "+reward_manager.name": "naive",
        "+reward_manager.source": "register",
        "algorithm.use_kl_in_reward": False,
        "algorithm.filter_groups.enable": True,
        "algorithm.filter_groups.metric": "seq_reward",
        "algorithm.filter_groups.criterion": "seq_reward",
        "algorithm.filter_groups.max_num_gen_batches": profile.get("filter_cap", 2),
        "algorithm.rdgdpo.w_max": args.w_max,
        "algorithm.rdgdpo.calibration": calibration or "positive",
        "custom_reward_function.path": str(ROOT/"runtime/reward.py"),
        "custom_reward_function.name": "compute_score",
        "data.train_files": str(data/"train.parquet"),
        "data.val_files": str(data/"test.parquet"),
        "data.train_batch_size": args.prompt_batch,
        "data.gen_batch_size": 3*args.prompt_batch//2,
        "data.val_batch_size": 128,
        "data.seed": args.seed,
        "data.max_prompt_length": 1024,
        "data.max_response_length": 8000,
        "data.filter_overlong_prompts": True,
        "actor_rollout_ref.model.path": model,
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.actor.optim.lr": 1e-6,
        "actor_rollout_ref.actor.ppo_mini_batch_size": ppo_batch,
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.use_dynamic_bsz": True,
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": profile.get("max_tokens_per_gpu", 16384),
        "actor_rollout_ref.actor.loss_agg_mode": "token-mean",
        "actor_rollout_ref.actor.clip_ratio_low": .2,
        "actor_rollout_ref.actor.clip_ratio_high": .28,
        "actor_rollout_ref.actor.use_kl_loss": True,
        "actor_rollout_ref.actor.kl_loss_coef": .0005,
        "actor_rollout_ref.actor.kl_loss_type": "mse",
        "actor_rollout_ref.actor.entropy_coeff": 0.,
        "actor_rollout_ref.actor.grad_clip": 1.,
        "actor_rollout_ref.actor.fsdp_config.model_dtype": "fp32",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": profile.get("optimizer_offload", False),
        "+actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype": "bf16",
        "+actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype": "fp32",
        "+actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype": "fp32",
        "actor_rollout_ref.actor.checkpoint.save_contents": ["model", "optimizer", "extra", "hf_model"],
        "actor_rollout_ref.actor.checkpoint.load_contents": ["model", "optimizer", "extra"],
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.seed": args.seed,
        "actor_rollout_ref.rollout.n": g,
        "actor_rollout_ref.rollout.temperature": 1.,
        "actor_rollout_ref.rollout.top_p": 1.,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": profile.get("tp", 1),
        "actor_rollout_ref.rollout.gpu_memory_utilization": profile.get("gpu_memory_utilization", .6),
        "actor_rollout_ref.rollout.max_num_batched_tokens": 32768,
        "actor_rollout_ref.rollout.max_num_seqs": profile.get("max_num_seqs", 256),
        "+actor_rollout_ref.rollout.val_kwargs.max_tokens": 8000,
        "actor_rollout_ref.ref.fsdp_config.param_offload": True,
        "actor_rollout_ref.ref.fsdp_config.model_dtype": "bf16",
        "trainer.logger": profile.get("logger", ["console"]),
        "trainer.project_name": "DARA-math",
        "trainer.experiment_name": name,
        "trainer.n_gpus_per_node": profile.get("gpus", 8),
        "trainer.nnodes": profile.get("nodes", 1),
        "trainer.save_freq": 10,
        "trainer.test_freq": 10,
        "trainer.default_local_dir": str(run_dir),
        "trainer.total_epochs": max(30, steps),
        "trainer.total_training_steps": steps,
        "trainer.resume_mode": "resume_path" if args.resume else "disable",
    }
    if static:
        overrides["+algorithm.rdgdpo.static_weights"] = [1., args.static_length_weight]
    if args.resume:
        overrides["trainer.resume_from_path"] = str(args.resume.expanduser().resolve())
    def hydra(value):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    cmd = [profile.get("python", "python3"), "-u", "-m", "verl.trainer.main_ppo"]
    cmd += [f"{key}={hydra(value)}" for key, value in overrides.items()]
    return cmd, framework, run_dir, overrides


def main():
    p = parser()
    args = p.parse_args()
    profile = json.loads(args.profile.read_text())
    try:
        cmd, framework, run_dir, overrides = build(args, profile)
        print(shlex.join(cmd), flush=True)
        if not args.execute:
            return
        if not (framework/"rdgdpo_runtime.json").is_file():
            raise ValueError("Prepare the paper runtime with scripts/prepare_runtime.py first")
        for key in ("data.train_files", "data.val_files"):
            if not Path(overrides[key]).is_file():
                raise ValueError(f"Missing dataset: {overrides[key]}")
        if run_dir.exists() and not args.resume:
            raise ValueError(f"Run directory exists: {run_dir}; use a full --resume checkpoint")
        if args.resume:
            checkpoint = args.resume.expanduser().resolve()
            if (not (checkpoint/"data.pt").is_file()
                    or not list((checkpoint/"actor").glob("optim_world_size_*_rank_*.pt"))
                    or not list((checkpoint/"actor").glob("extra_state_world_size_*_rank_*.pt"))):
                raise ValueError("Resume needs global_step_N with data.pt and actor optimizer/extra-state shards; HF weights alone are insufficient")
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_name = "resume_command.json" if args.resume else "command.json"
        (run_dir/manifest_name).write_text(json.dumps(overrides, indent=2)+"\n")
        env = os.environ.copy()
        env.update({k: str(v) for k, v in profile.get("environment", {}).items()})
        ray_tmp = Path(profile.get("ray_tmp_root", "/tmp")).expanduser().resolve()
        ray_tmp /= f"rdgdpo-ray-{hashlib.sha256(str(run_dir).encode()).hexdigest()[:12]}"
        env.update(PYTHONPATH=str(framework), SEED=str(args.seed), LENGTH_LIMIT="4000",
                   EXPERIMENT_NAME="qwen-"+run_dir.name, RAY_ADDRESS="local",
                   RAY_TMPDIR=str(ray_tmp), TOKENIZERS_PARALLELISM="false",
                   REFINEDREWARD="0", COARSEREWARD="0", INTERMEDIATEREWARD="0")
        result = subprocess.run(cmd, cwd=framework, env=env)
        sys.exit(result.returncode)
    except (ValueError, OSError) as error:
        p.exit(2, str(error)+"\n")


if __name__ == "__main__":
    main()
