# 当前实验状态

截至本次交接，共整理 Haotian 下42个完整100step训练run及两个base的最终评测，44个模型均有当前V3两组AST和V4三组结果。现有结果全部为correctness+format两奖励；新三奖励矩阵尚未执行。

| 模型 | 方法 | 已完成 training seeds | Final BFCL V3/V4 |
|---|---|---|---|
| 1.5B | GRPO、GDPO、DARA | 各0、1、2、4、5，共15run | 已完成 |
| 1.5B | RVPO、DVAO、GDPO-SAW、GD²PO-Hard | 各0、1、2，共12run | 已完成 |
| 3B | GRPO、GDPO、DARA | 各0、1、2、4、5，共15run | 已完成 |
| 3B | RVPO、DVAO、GDPO-SAW、GD²PO-Hard | 尚未训练 | 没有对应训练后结果 |
| 1.5B / 3B | Base | 不适用 | 各一份固定协议结果 |

完整数值见 [ALL_SEEDS.md](../results/reference/ALL_SEEDS.md) 和 [per_model.csv](../results/reference/per_model.csv)。核心结果如下，均为五seed、百分数、mean±sample SD：

| Size | Method | V3 Live AST | V3 Non-Live AST | V4 Avg Acc | V4 Avg Format |
|---|---|---:|---:|---:|---:|
| 1.5B | GDPO | 66.91±2.67 | 79.54±2.27 | 50.37±1.12 | 97.11±1.00 |
| 1.5B | DARA | 68.76±1.19 | 80.26±1.11 | 51.17±0.78 | 97.55±0.64 |
| 3B | GDPO | 69.34±1.05 | 82.95±0.92 | 53.66±0.96 | 96.57±0.77 |
| 3B | DARA | 70.41±1.06 | 82.98±0.84 | 54.15±0.58 | 97.22±0.66 |

Training dynamics按逐步reward单独绘图。1.5B训练format首次达到0.8的step：DARA为14/16/15/13/29，平均17.4；GDPO为18/19/75/16/65，平均38.6；GRPO为37/43/34/31/31，平均35.2。原始逐步数据可自行重画。

## 已观察到的基线问题

DVAO seed1、GD²PO-Hard seed2跑完100steps，但format未收敛，末次训练内validation format分别为0.1125和0.0875。其BFCL AST正确率并未同幅崩溃，所以异常主要体现为format表现；三个seed的V4 Avg Format分别为69.02±44.42、67.98±48.58。这两个seed的format表现拉大了跨seed标准差。

完整表使用全部seed。[SELECTED_SEEDS.md](../results/reference/SELECTED_SEEDS.md) 另外报告 DVAO seeds0/2 和 GD²PO-Hard seeds0/1，各n=2，并标明筛选依据。

RVPO三seed均完成且达到format收敛，现有最终评测总体正常；一次历史seed2出现过很大的有限grad-norm数值，日志已保留。

GDPO-SAW在当前工具调用任务表现很强，与DARA的优势差距较小；用户决定本轮后续工具调用消融比较聚焦GRPO、GDPO、DARA，暂不把SAW列入这两组新增矩阵。仓库保留SAW实现及全部三seed结果。

## 新实验的完成状态

Group消融G4可复用核心三个方法在1.5B的15个完整run。G8/16/32的45个run尚待完整执行。历史GRPO G8 seed0停在step43，该配置按完整100step重新运行。

三奖励GRPO/GDPO/DARA × 1.5B/3B × 五seed共30个run待执行。代码与实验清单已准备，执行状态为待运行。
