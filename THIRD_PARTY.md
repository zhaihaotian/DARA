# Source provenance

| Component | Source and scope |
|---|---|
| Haotian runtime | https://github.com/zhaihaotian/RD-GDPO ; copied from the actual training worktree, including the locally migrated baselines, on 2026-09-17 |
| GDPO / verl lineage | https://github.com/NVlabs/GDPO ; the Haotian old verl/GDPO infrastructure is retained |
| Baseline comparison source | https://github.com/JayCheng113/RD-GDPO ; upstream comparison point 917da15 and the previously used local runtime |
| ToolRL data recipe and BFCL handler | https://github.com/qiancheng0/ToolRL ; pinned download of the original rlla_4k parquet and adapted RLLA evaluation handler |
| BFCL official evaluator/data | https://github.com/ShishirPatil/gorilla ; V3 package2025.8.6.2 and V4 commit6ea57973c7a6097fd7c5915698c54c17c5b1b6c8 |
| FlashAttention wheel | https://github.com/Dao-AILab/flash-attention/releases/tag/v2.6.3 |

The root Apache-2.0 license and original source copyright notices are retained from the training repository. Upstream BFCL is installed separately with its own notices and dataset documentation. Algorithm and infrastructure distinctions are documented in docs/ALGORITHMS.md. The training-data download preserves the exact existing ToolRL-derived experimental split. Training parquet, model weights and BFCL datasets are obtained separately; their sources are documented.
