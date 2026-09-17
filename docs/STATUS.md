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

| Size | Method | V3 Live AST | V3 Non-Live AST | V4 Avg Acc | V4 Avg Format* |
|---|---|---:|---:|---:|---:|
| 1.5B | GDPO | 66.91±2.67 | 79.54±2.27 | 50.37±1.12 | 97.11±1.00 |
| 1.5B | DARA | 68.76±1.19 | 80.26±1.11 | 51.17±0.78 | 97.55±0.64 |
| 3B | GDPO | 69.34±1.05 | 82.95±0.92 | 53.66±0.96 | 96.57±0.77 |
| 3B | DARA | 70.41±1.06 | 82.98±0.84 | 54.15±0.58 | 97.22±0.66 |

这些数值是现有五seed的描述，不能推广成所有指标的方差都更小；例如3B的V3 Live SD约1.06和1.05几乎相同。Training dynamics单独画图，BFCL表不放收敛step、最慢seed或best checkpoint。1.5B训练format首次达到0.8的step：DARA为14/16/15/13/29，平均17.4；GDPO为18/19/75/16/65，平均38.6；GRPO为37/43/34/31/31，平均35.2。原始逐步数据可自行重画。

## 已观察到的基线问题

DVAO seed1、GD²PO-Hard seed2跑完100steps，但format未收敛，末次训练内validation format分别为0.1125和0.0875。其BFCL AST正确率并未同幅崩溃，所以异常主要体现为format表现；三个seed的V4 Avg Format*分别为69.02±44.42、67.98±48.58。这种大SD来自真实seed间差异，不是把标准差单位算错，也不能直接说整体任务学习完全失败。

应保留全部seed作为完整实验记录。用户此前查看的去除这两个seed后的结果单独保存在 [SELECTED_SEEDS.md](../results/reference/SELECTED_SEEDS.md)，标明事后筛选、每方法剩余n=2。不能把筛选后的结果标成3-seed无条件性能，不能静默删日志、换seed或重新挑checkpoint。

RVPO三seed均完成且达到format收敛，现有最终评测总体正常；一次历史seed2出现过很大的有限grad-norm数值，日志已保留，不能由“最终正常”推断每一步都无数值异常。

GDPO-SAW在当前工具调用任务表现很强，与DARA的优势差距较小；用户决定本轮后续工具调用消融比较聚焦GRPO、GDPO、DARA，暂不把SAW列入这两组新增矩阵。这是研究比较范围的决定，不是SAW失败或实现不可用。仓库保留SAW实现及全部三seed结果；对外不能据缩小后的比较范围声称优于所有已测试方法。

## 新实验的完成状态

Group消融G4可复用核心三个方法在1.5B的15个完整run。G8/16/32的45个run尚待完整执行。历史存在一个GRPO G8 seed0在step43停止的run，不计完成，也不将其混入完整100step结果。

三奖励GRPO/GDPO/DARA × 1.5B/3B × 五seed共30个run待执行。新接线已准备；完整100step GPU训练效果尚无结果。交接的CPU数值回归和BFCL重新评分验证不等于完成这些新训练。
