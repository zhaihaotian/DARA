# 3B DVAO and GD²PO-Hard seed-completion results

This directory contains the completed four-run 8×A100 40GB rental experiment documented in [`operations/RUNBOOK.md`](operations/RUNBOOK.md). The runs use Qwen2.5-3B-Instruct with DVAO or GD²PO-Hard, seeds 1 and 2, group size 4, correctness plus format rewards, four ranks per run, 100 training steps, and checkpoints saved every 10 steps. All four runs reached step 100, all 40 checkpoints were saved, and every checkpoint completed BFCL V4 evaluation. The four step100 evaluations are also the final results.

## Training dynamics

[`training/training_dynamics.csv`](training/training_dynamics.csv) combines all 404 run-step rows and retains every column written by the original per-run training exports. [`training/training_runs.csv`](training/training_runs.csv) records method, seed, rank allocation, start time, elapsed time, row counts, and saved steps. [`training/checkpoint_inventory.csv`](training/checkpoint_inventory.csv) records the remote source path, byte size, and file count for all 40 checkpoints without copying model weights or optimizer state.

Each directory under [`training/runs`](training/runs) contains the original `metrics.jsonl`, original `training_dynamics.csv`, expanded config, launch settings, command settings, runtime metadata, elapsed time, exit codes, and the checkpoint-local `config.json` at steps 10, 20, ..., 100. No checkpoint weights are included.

## BFCL V4 evaluation

The [`evaluation`](evaluation) directory contains the four completed BFCL V4 exports. `checkpoint_per_model.csv` has one row for every run-step pair. `checkpoint_method_results.csv` contains the mean and sample standard deviation across seeds 1 and 2 for each method and step. `process_final_per_model.csv` contains the four step100 rows, and `process_final_method_results.csv` contains the two method-level final aggregates. `process_bfcl_v4_table.tex` and `final_bfcl_v4_table.tex` render these process and final results directly.

Evaluation covers 3,301 BFCL V4 cases per checkpoint. The fixed protocol uses the ToolRL prompt and parser, temperature 0.6, top-p 0.95, inference seed 0, an 8,192-token generation limit, and the BFCL V4 source at gorilla commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`. Accuracy and Format use the aggregation rules recorded in the runbook.

The private Hugging Face data mirror is [`zhaihaotian/rd-gdpo-experiment-results`](https://huggingface.co/datasets/zhaihaotian/rd-gdpo-experiment-results/tree/main/rental_3b_20260920). Its [`bfcl_v4_raw_archives`](https://huggingface.co/datasets/zhaihaotian/rd-gdpo-experiment-results/tree/main/rental_3b_20260920/bfcl_v4_raw_archives) directory contains one archive per run with all ten checkpoints' raw inference, official scores, per-case diagnostics, summaries, inference metadata, and logs. [`evaluation/bfcl_v4_archives.csv`](evaluation/bfcl_v4_archives.csv) records each archive name, run, Hugging Face path, and verified byte size. The four archives total 208,964,065 bytes and contain no checkpoint weights or optimizer state.

## Completion and verification

The controller completed with `reason=all_complete` at `2026-09-20T08:38:13.545905+00:00`. [`operations/final_status.json`](operations/final_status.json) records 4/4 trainings, 40/40 checkpoint evaluations, and 4/4 final evaluations. [`operations/queue_state.json`](operations/queue_state.json) retains the full final controller queue state. [`verification.json`](verification.json) checks the run, training-row, checkpoint inventory, method-step aggregate, final-evaluation, metadata, and unique-key counts.

Rebuild the combined training files and repeat the local checks with:

```bash
python build_results.py
```

The one-time inventory refresh reads sizes and file counts from the remote checkpoints without copying them:

```bash
python build_results.py --refresh-inventory-host ubuntu@137.131.62.225
```
