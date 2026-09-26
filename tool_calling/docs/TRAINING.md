# Training configuration

Every method is trained with the configuration below. Only the advantage estimator (`--method`) differs between runs in a comparison. The data are the ToolRL `rlla_4k` parquet files from `scripts/fetch_data.py`, with the original prompts, ground truth, order and split: 3,920 training and 80 validation examples.

| Parameter | Value |
|---|---|
| Backbone | Qwen2.5-1.5B-Instruct / Qwen2.5-3B-Instruct |
| Hardware per run | 1 node, 4 × NVIDIA A100 40GB |
| Training / rollout | FSDP / vLLM 0.6.3, tensor parallel 1, bf16 |
| Python / PyTorch / Transformers / Ray | 3.10.21 / 2.4.0+cu121 / 4.47.1 / 2.10.0 |
| FlashAttention / xFormers | 2.6.3 / 0.0.27.post2 |
| Training steps | 100 |
| Validation | before training and every 10 steps (80 RLLA examples) |
| Evaluated checkpoint | step 100 (Hugging Face actor and tokenizer) |
| Prompts × responses per step | 512 × 4 = 2,048 responses (G = 4) |
| Responses per PPO mini-batch | 512 (4 optimizer updates per step) |
| PPO epochs / clip ratio | 1 / 0.2 |
| Learning rate / gradient clipping | 1e-6 / 1.0 |
| Entropy coefficient | 0.001 |
| KL | reward-KL penalty with coefficient 0.001; no separate actor KL loss |
| Rollout temperature / top-p / top-k | 1.0 / 1.0 / −1 |
| Prompt / response limit | 2,048 / 1,024 tokens |
| Dynamic batching | on; 16,384 tokens per GPU |
| Rollout GPU memory utilization | 0.6 |
| Actor param / grad / optimizer offload | off / off / off |
| Reference param offload | on |
| Gradient checkpointing / padding removal | on / on |
| DARA weight cap $w_{\max}$ | 5 |
| Seeds | {0, 1, 2, 4, 5} for the main comparison; {0, 1, 2} for the rollout-group and three-reward analyses; the vLLM rollout engine seed is 0 |

GRPO computes advantages from the summed reward after the reward-KL penalty. The per-channel methods (GDPO, DVAO, GD²PO-Hard, DARA-Asym, DARA-Sym) read the individual, unpenalized reward channels. The policy loss uses dynamic micro-batch accumulation: each micro-batch token-mean loss is divided by the nominal accumulation factor of 2. For 3B runs the launcher sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. A 1.5B run takes roughly 2.5 hours on 4 × A100 40GB, and a 3B run 4–5 hours.

## Rewards

The two base rewards are the GDPO/ToolRL ones, computed by `vendor/verl/verl/utils/reward_score/rlla.py`:

| Channel | Range | Definition |
|---|---|---|
| correctness | [−3, 3] | tool name, parameter-name and parameter-value match against the ground-truth calls |
| format | {0, 1} | the output exactly matches one of the `<think>…</think>` + `<tool_call>` / `<response>` structures |
| length (`--rewards three`) | {0, 1} | $\mathbb{1}[n \le N]$ with $N$ = `--length-max-words` (16), where $n$ is the whitespace word count of the last `<think>` span. Missing think tags score 0. |

In the three-reward setting, GRPO sums all three channels. GDPO and both DARA variants standardize each of the three channels separately, and DARA takes $\pi_{\mathrm{ref}}$ as the maximum over the three densities. 

## Group-size ablation budget

| G | Prompts per step | PPO mini-batch (prompts) | PPO micro-batch (prompts) | Responses per step |
|---:|---:|---:|---:|---:|
| 4 | 512 | 128 | 64 | 2,048 |
| 8 | 256 | 64 | 32 | 2,048 |
| 16 | 128 | 32 | 16 | 2,048 |
| 32 | 64 | 16 | 8 | 2,048 |

This design fixes the number of generated responses and optimizer updates per step. A larger G therefore trades distinct prompts for more responses per prompt. All other settings are unchanged.

## Outputs

Each run directory contains `launch.json` (launcher arguments and resolved settings), `command.json`, the full Hydra `config.json`, `metrics.jsonl` (one record per step), `exit.code` and `actor/global_step_100`. With `--save-freq 10`, checkpoints are also written at steps 10, 20, …, 90. These are HF weights for evaluation; optimizer and data-loader state are not saved. An interrupted run is restarted from the base model in a new output directory with the same seed.

`metrics.jsonl` retries a write up to 30 times on a filesystem `EIO` error, so transient storage faults on shared filesystems do not end a run.
