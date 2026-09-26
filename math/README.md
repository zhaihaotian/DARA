# DARA on Math Reasoning

This folder reproduces the math-reasoning experiments in the paper. The model has to solve competition math problems under a length budget, with two binary rewards:
- **Correctness**: 1 if the final `\boxed{}` answer matches the reference.
- **Length**: 1 if the response is at most 4,000 tokens.

Early in training almost every rollout group has both short and long responses. Later on, as the model learns to stay under the limit, the length reward becomes sparse. That late phase is where DARA helps most.

We follow the math setup of GDPO (§4.2): DeepScaleR-Preview data, DeepSeek-R1 prompt format and DAPO-style training. The training stack combines verl v0.7 with the multi-reward advantage estimators, per-reward logging, dynamic sampling and exact resume implemented in [`runtime/overlay/`](./runtime/overlay).

## Installation

```bash
bash scripts/bootstrap_runtime.sh --prefix /path/to/dara-math-v07
```

The script:
1. creates a Python 3.12 venv with the pinned dependencies from [`requirements-runtime-lock.txt`](./requirements-runtime-lock.txt);
2. clones verl `v0.7.0` (commit `f9c855f7`) and installs FlashAttention;
3. copies our overlay on top, checking it against [`runtime/overlay.sha256`](./runtime/overlay.sha256);
4. runs a GPU preflight check.

If you already have a verl 0.7 checkout, you can do only step 3:

```bash
python3 scripts/prepare_runtime.py --base-framework /path/to/verl-v0.7.0 --output /path/to/framework-paper
```

Then copy [`configs/machine.example.json`](./configs/machine.example.json) to e.g. `my_machine.json`, and fill in:
- `python`: the venv interpreter;
- `framework`: the assembled runtime;
- `output_root`: where runs are written;
- `models`: local paths to the backbones.

The example rollout settings are the ones we used for 1.5B: memory fraction 0.9, 256 concurrent sequences, and up to 2 generation batches for dynamic sampling. For the 4B models we used 0.8, 160 and 4.

## Data

The DeepScaleR-Preview training set is already in [`data/deepscaler/train.parquet`](./data/deepscaler). `test.parquet` holds the 200 seeded examples we use only for online monitoring. Prompts longer than 1,024 tokens are dropped at load time.

## Training

```bash
python3 scripts/train.py --model r1-1.5b --method dara_asym --seed 0 --profile my_machine.json --execute
```

Without `--execute`, the script only prints the full verl command, which is handy for checking a config.

- **Models** (`--model`): `r1-1.5b` (DeepSeek-R1-Distill-Qwen-1.5B), `qwen3-instruct` (Qwen3-4B-Instruct-2507), `qwen3-thinking` (Qwen3-4B-Thinking-2507) and `r1-7b` (DeepSeek-R1-Distill-Qwen-7B).
- **Methods** (`--method`): `grpo`, `gdpo`, `dvao`, `gd2po_hard`, `dara_asym` (our main method) and `dara_sym`.
- **Fixed-weight ablation**: `--method static_asym` / `static_sym` keeps the DARA-Asym / DARA-Sym pipeline but uses fixed weights, correctness 1 and length 5 (change the 5 with `--static-length-weight`).

The main shared settings:
- **Batch**: 512 accepted prompt groups × 16 rollouts per step, drawn from candidate batches of 768 groups. A group is kept only if its rewards vary.
- **Optimization**: learning rate 1e-6, PPO mini-batch 64, clip 0.2 / 0.28, MSE KL loss 5e-4, token-mean loss.
- **Lengths**: up to 8,000 response tokens.
- **Checkpoints**: model, optimizer and data state every 10 updates.

Everything is written out in [`scripts/train.py`](./scripts/train.py) and matches Table 9 of the paper. To continue an interrupted run, pass `--resume <run>/global_step_N` with the same method, seed and `--steps`.

Some examples:

```bash
# all methods on DeepSeek-R1-1.5B
for m in grpo gdpo dvao gd2po_hard dara_asym dara_sym; do
  python3 scripts/train.py --model r1-1.5b --method $m --seed 0 --profile my_machine.json --execute
done

# fixed-weight ablation
python3 scripts/train.py --model r1-1.5b --method static_asym --seed 0 --profile my_machine.json --execute

# Instruct vs. Thinking, trained to step 100
python3 scripts/train.py --model qwen3-thinking --method dara_asym --seed 0 --steps 100 --profile my_machine.json --execute
```

The per-step logs include:
- `rollout_prefilter/{correctness,length}/mean`: rewards on the full rollout pool before filtering;
- `rdgdpo/pi_*` and `rdgdpo/w_*`: the densities and weights DARA uses. `rdgdpo` is the internal name of the DARA estimator.

## Evaluation

**Held-out benchmarks.** We evaluate on MATH-500, AIME 2024, AMC 2022/2023, Minerva and OlympiadBench (English competition subset): 1,559 questions, 16 samples each, temperature 0.6, top-p 0.95, up to 32,768 tokens.

```bash
PYTHONPATH=/path/to/framework-paper /path/to/venv/bin/python evaluation/eval_math.py \
  --ckpt <run>/global_step_50/actor/huggingface --bench all --n 16 --tp 2 --seed 0 --out eval/math.json
```

The script reports:
- **Acc**: pass@1 averaged over the 16 samples, using the strict boxed-answer grader with a numeric-equivalence fallback.
- **Exceed**: the fraction of responses longer than 4,000 tokens.
- **Joint**: the fraction that is both correct and within the limit.

Truncated responses count as wrong. All generations are saved next to the output, so you can rescore without the GPU.

**Training-side length compliance.** For the convergence curves, we look for the first step at which the trailing 10-step mean of `rollout_prefilter/length/mean` crosses a threshold and stays above it for the next 20 steps:

```bash
python3 tools/learning_efficiency.py history.csv --label 1.5b-dara_asym-s0 --budget 200 --threshold 0.99
```

## Sanity check

```bash
python3 tools/check_advantages.py
```

This is a quick CPU test of the estimators. Among other things, it checks that DARA with all weights equal to 1 is exactly GDPO, and that the Static variants reduce to the right formulas.
