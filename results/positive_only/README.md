# RD-GDPO positive-only results

This directory contains the Haotian-infrastructure results for the single-sided density calibration baseline, recorded as `rdgdpo_positive`. The method multiplies positive channel advantages by the density weight and leaves negative channel advantages unchanged.

| File | Contents |
|---|---|
| `final_per_model.csv` | Final BFCL scores for every evaluated seed |
| `final_method_results.csv` | Mean and sample standard deviation across seeds |
| `training_dynamics.csv` | Per-step reward, validation, density, weight, and active-group metrics |
| `training_status.csv` | Completion state, first format-reward step at or above 0.8, and final metrics |
| `training_logs/` | Original `metrics.jsonl` for each training run |

The 1.5B final comparison contains seeds 0, 1, 2, 4, and 5. BFCL-V3 reports Live and Non-Live AST accuracy; BFCL-V4 reports Live, Non-Live, Multi-Turn, and their arithmetic Average, together with the corresponding Format metrics. The 3B V4 comparison contains seeds 0, 1, 2, and 4. Its seed 5 training record ends at step 76 and is retained in the training files without entering the final BFCL aggregation.

The process-checkpoint replay selects 1.5B seeds 0, 4, and 5 by their final BFCL-V4 Average Accuracy. Each replay saves steps 10, 20, ..., 100.
