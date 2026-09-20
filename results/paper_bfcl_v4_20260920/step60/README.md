# BFCL V4：1.5B step 60

| Size | Method | Live Acc. | Live Format | Non-Live Acc. | Non-Live Format | Multi-Turn Acc. | Multi-Turn Format | Average Acc. | Average Format |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.5B | Base | 9.33 | 0.37 | 12.06 | 0.12 | 0.38 | 0.21 | 7.25 | 0.23 |
| 1.5B | GRPO | 63.26 ± 0.21 | 99.78 ± 0.13 | 77.39 ± 0.75 | 99.88 ± 0.09 | 2.38 ± 0.22 | 24.48 ± 1.00 | 47.68 ± 0.16 | 74.71 ± 0.28 |
| 1.5B | GDPO | 68.21 ± 0.37 | 99.96 ± 0.05 | 79.84 ± 0.43 | 100.00 ± 0.00 | 3.44 ± 0.62 | 64.28 ± 16.72 | 50.50 ± 0.47 | 88.08 ± 5.59 |
| 1.5B | DVAO | 68.10 | 99.85 | 79.54 | 100.00 | 2.25 | 57.04 | 49.96 | 85.63 |
| 1.5B | GD²PO-Hard | 67.14 | 100.00 | 79.81 | 100.00 | 5.12 | 70.28 | 50.69 | 90.09 |
| 1.5B | DARA | 68.37 ± 0.82 | 99.98 ± 0.04 | 79.84 ± 0.22 | 100.00 ± 0.00 | 4.62 ± 0.12 | 88.19 ± 3.48 | 50.94 ± 0.32 | 96.06 ± 1.17 |

数值为百分数，跨 seed 计算 mean ± sample SD（ddof=1）。Average 先对每个 run 的 Live、Non-Live、Multi-Turn 取算术平均，再跨 run 汇总。Base 为训练前模型。

1.5B：GRPO seeds 0/2/5，GDPO 0/1，DVAO 3，GD²PO-Hard 4，DARA 0/1/2。3B：DVAO 与 GD²PO-Hard 均为 seeds 0/1/2。DVAO 与 GD²PO-Hard 的 1.5B 结果各来自一个 seed，显示单次分数。

纳入条件为当前 checkpoint 的收敛状态：step60 RLLA validation Format ≥0.8，同样应用到所有训练方法。读取每个 run 的 step60 训练日志后筛选，共21个候选checkpoint，保留16个。selection.csv 保存逐 checkpoint 的日志指标、纳入状态及排除原因。

图表展示1.5B的Base及五个方法。CSV保留1.5B和3B全部现有step60数据，可按model_size筛选。

本表来自最新 PR #1（a954110）的过程评测，以及本地已完成的 3B DVAO / GD²PO-Hard seed0 过程评测。上级 inputs 目录保留本地数据快照。per_run.csv 为逐 run 结果，method_results.csv 为统计表，coverage.csv 为各方法 seed 覆盖情况。

生成命令：`python results/paper_bfcl_v4_20260920/step60/build_table.py`。脚本依赖上级 build_figures.py、CSV、NumPy 和 Matplotlib。LaTeX 使用 booktabs。
