# DARA on Tool Calling

This folder reproduces the tool-calling experiments in the paper. The model learns to call tools from a user query and a set of tool descriptions, answering in this format:

```
<think> reasoning </think>
<tool_call>
{"name": ..., "parameters": {...}}
</tool_call>
<response> final answer to the user </response>
```

Training uses two rewards, the same as GDPO:
- **Format**: 0 or 1, checking the required structure.
- **Correctness**: in [−3, 3], matching tool names, parameter names and parameter values against the reference calls.

The code is the verl implementation released with [GDPO](https://github.com/NVlabs/GDPO), vendored in [`vendor/verl`](./vendor/verl), with DARA and the baselines added as advantage estimators. All methods share the same launcher and configuration.

## Installation

We use three separate conda environments, because the training stack and the BFCL evaluation need different versions of vLLM and PyTorch. All versions are pinned in [`environments/`](./environments).

```bash
bash environments/install_training.sh dara-train          # training: PyTorch 2.4 + vLLM 0.6.3
bash environments/install_inference.sh dara-inference     # vLLM 0.11 server used during evaluation
bash environments/install_evaluation.sh dara-bfcl-v4      # BFCL-v4 (gorilla @ 6ea57973)
```

The training installer finishes by running the CPU tests (`bash scripts/test.sh`).

## Data

```bash
python scripts/fetch_data.py
```

This downloads ToolRL's `rlla_4k` split (3,920 training and 80 validation examples) at a pinned commit into `data/rlla_4k/`. The models are `Qwen/Qwen2.5-1.5B-Instruct` and `Qwen/Qwen2.5-3B-Instruct`; download them wherever you like and pass the path.

## Training

Each run uses one node with 4× A100 40GB and trains for 100 steps (512 prompts × 4 rollouts per step). The final model is saved to `<output>/actor/global_step_100`.

```bash
conda activate dara-train
python training/launch.py --method dara_asym --seed 0 --model-size 1.5b \
  --model /models/Qwen2.5-1.5B-Instruct --output outputs/1p5b-dara_asym-g4-s0
```

`--method` can be `grpo`, `gdpo`, `dvao`, `gd2po_hard`, `dara_asym` or `dara_sym`. For 3B, use `--model-size 3b` with the 3B model path. Add `--dry-run` to print the full resolved config without training. Add `--save-freq 10` to also keep the checkpoints at steps 10, 20, … (we used these for the checkpoint-wise BFCL curves).

The main comparison uses seeds 0, 1, 2, 4 and 5:

```bash
for m in grpo gdpo dvao gd2po_hard dara_asym dara_sym; do
  for s in 0 1 2 4 5; do
    python training/launch.py --method $m --seed $s --model-size 1.5b \
      --model /models/Qwen2.5-1.5B-Instruct --output outputs/1p5b-$m-g4-s$s
  done
done
```

**Rollout-group analysis.** Here G varies over 4, 8, 16 and 32, with the number of responses per step fixed at 2,048 (so fewer prompts per step for larger G). The 48 runs (GRPO, GDPO, DARA-Asym and DARA-Sym, seeds 0–2) are listed in [`configs/group_ablation.json`](./configs/group_ablation.json). Run any of them by id:

```bash
python training/run_manifest.py --manifest configs/group_ablation.json \
  --id 1p5b-dara_asym-g16-s0-two --model-root /models --output-root outputs
```

**Three rewards.** This setting adds a third reward: 1 if the `<think>` block has at most 16 words (`--rewards three --length-max-words 16`). The 12 runs are listed in [`configs/length_le16_binary.json`](./configs/length_le16_binary.json):

```bash
python training/run_manifest.py --manifest configs/length_le16_binary.json \
  --id 1p5b-dara_asym-g4-s0-three-len16 --model-root /models --output-root outputs
```

Every run writes a `metrics.jsonl` with one line per step. The per-reward training rewards are `critic/{correctness,format,length}_score/mean`. The validation scores are `val/test_*/rlla`, logged every 10 steps. DARA's densities and weights are `dara/pi_*`, `dara/w_*` and `dara/active_groups_*`. It loads straight into pandas:

```python
import pandas as pd
df = pd.read_json("outputs/1p5b-dara_asym-g4-s0/metrics.jsonl", lines=True)
```

## Evaluation

We evaluate on BFCL-v4 (Live, Non-Live and Multi-Turn; 3,301 cases), with one GPU per checkpoint. The script starts a vLLM server from the inference environment, runs every case through the ToolRL prompt and parser, and then scores it with the official BFCL checkers.

```bash
conda activate dara-bfcl-v4
CUDA_VISIBLE_DEVICES=0 python evaluation/run.py \
  --model outputs/1p5b-dara_asym-g4-s0/actor/global_step_100 \
  --server-python /path/to/envs/dara-inference/bin/python \
  --port 8000 --output eval/v4/1p5b-dara_asym-g4-s0
```

Decoding follows the paper: temperature 0.6, top-p 0.95, at most 8,192 new tokens, seed 0. For three-reward checkpoints, add `--length-max-words 16` to also report length compliance. If the run gets interrupted, rerun the same command; it only generates the missing cases. To rescore saved outputs on CPU:

```bash
python evaluation/score.py --output eval/v4/1p5b-dara_asym-g4-s0
```

The numbers end up in `summary.json` under `groups` (accuracy, format and, if enabled, length, for live / non_live / multi_turn / average). `diagnostics_per_case.csv` has the per-case values. We report the mean over training seeds. [`docs/EVALUATION.md`](./docs/EVALUATION.md) spells out exactly how Format and the group averages are computed.

## More details

- [`docs/TRAINING.md`](./docs/TRAINING.md): the full training configuration and the batch sizes of the rollout-group study.
- [`docs/ALGORITHMS.md`](./docs/ALGORITHMS.md): how each method computes its advantage in this code base.
- [`docs/EVALUATION.md`](./docs/EVALUATION.md): the BFCL-v4 protocol and metric definitions.
