# 1.5B rental checkpoint study results

This directory contains the completed 8×A100 40GB rental experiment defined by [`configs/rental_1p5b_20h.json`](../../configs/rental_1p5b_20h.json). All 15 runs reached 100 training steps and saved steps 10, 20, ..., 100. All 150 checkpoints were evaluated with the repository's fixed BFCL V4 protocol, and all 15 step100 checkpoints are included in the final table.

## Training dynamics

[`training/training_dynamics.csv`](training/training_dynamics.csv) combines all 1,515 run-step rows and keeps the reward, validation, density, optimization, sequence-length, and timing fields written by training. [`training/training_runs.csv`](training/training_runs.csv) records the method, seed, hardware ranks, elapsed time, and saved steps for each run. [`training/checkpoint_inventory.csv`](training/checkpoint_inventory.csv) lists the source path, byte size, and file count of all 150 HF checkpoints. The per-run directories retain the original `metrics.jsonl`, `training_dynamics.csv`, expanded config, launch settings, command settings, and runtime metadata.

The process study uses three fixed seeds per method: GRPO 0/2/5, GDPO 0/1/5, DARA 0/1/2, DVAO 3/4/5, and GD²PO-Hard 0/4/5. These are the seeds specified by the rental manifest and documented in [`docs/RENTAL_1P5B_20H.md`](../../docs/RENTAL_1P5B_20H.md).

## BFCL V4 evaluation

The `evaluation` directory contains the four requested exports. `checkpoint_per_model.csv` has one row for every run-step pair. `checkpoint_method_results.csv` contains the mean and sample standard deviation across the three seeds for each method and step. `process_final_per_model.csv` contains the 15 step100 rows, while `process_final_method_results.csv` contains the five method-level final aggregates. The two LaTeX files render the process and final tables directly from these results.

Evaluation covers the 14 BFCL V4 categories and 3,301 cases per checkpoint. It uses the ToolRL prompt and parser, temperature 0.6, top-p 0.95, inference seed 0, an 8,192-token generation limit, and the BFCL V4 source at gorilla commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`. Accuracy and Format follow the aggregation rules in [`docs/EVALUATION.md`](../../docs/EVALUATION.md).

## Completion and verification

The controller completed with `reason=all_complete` at `2026-09-19T18:11:23.275409+00:00`. [`operations/final_status.json`](operations/final_status.json) records 15/15 trainings, 150/150 checkpoint evaluations, and 15/15 final evaluations. [`verification.json`](verification.json) checks the run, row, seed, and unique run-step counts. Rebuild the combined training files and repeat these checks with:

```bash
python results/rental_1p5b_20260919/build_results.py
```

The private Hugging Face data mirror is [`zhaihaotian/rd-gdpo-experiment-results`](https://huggingface.co/datasets/zhaihaotian/rd-gdpo-experiment-results/tree/main/rental_1p5b_20260919). Its [`bfcl_v4_raw_archives`](https://huggingface.co/datasets/zhaihaotian/rd-gdpo-experiment-results/tree/main/rental_1p5b_20260919/bfcl_v4_raw_archives) directory contains one compressed archive per training run with the raw inference, official scores, per-case diagnostics, and summaries for all ten checkpoints. [`evaluation/bfcl_v4_archives.csv`](evaluation/bfcl_v4_archives.csv) records the 15 archive names and sizes. Git contains the compact, reviewable, directly plottable data.
