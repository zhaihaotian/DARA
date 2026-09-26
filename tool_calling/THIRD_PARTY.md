# Source provenance

| Component | Source and scope |
|---|---|
| Training runtime | https://github.com/zhaihaotian/RD-GDPO ; the verl/GDPO training worktree used for all reported runs, including the migrated baselines |
| GDPO / verl lineage | https://github.com/NVlabs/GDPO ; the Haotian old verl/GDPO infrastructure is retained |
| Baseline reference implementations | https://github.com/JayCheng113/RD-GDPO ; upstream comparison point 917da15 |
| ToolRL data recipe and BFCL handler | https://github.com/qiancheng0/ToolRL ; pinned download of the original rlla_4k parquet and adapted RLLA evaluation handler |
| BFCL official evaluator/data | https://github.com/ShishirPatil/gorilla ; BFCL V4 at commit 6ea57973c7a6097fd7c5915698c54c17c5b1b6c8 |
| FlashAttention wheel | https://github.com/Dao-AILab/flash-attention/releases/tag/v2.6.3 |

The Apache-2.0 license and original source copyright notices are retained. Upstream BFCL is installed separately with its own notices and dataset documentation. The estimators are documented in docs/ALGORITHMS.md. The training-data download reproduces the exact ToolRL rlla_4k split. Training parquet files, model weights and BFCL datasets are obtained separately from the sources above.
