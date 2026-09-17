# 1.5B DVAO seed0 复跑对照

原始训练在step60首次达到训练Format0.8，最终为0.9526。本轮每10steps保存模型的复跑最终Format为0.0713，最高0.0996；最终Correctness reward为1.8834。两次均完整训练100steps，seed为0，资源为四张A100。

原始训练在aga13，本轮复跑在agb02。两个展开配置中的模型路径、batch、G、优化器、学习率、rollout设置、DVAO权重与epsilon一致。数据路径分别指向原仓库与DARA，两个train/test parquet文件逐字节相同。配置差异为保存间隔100→10、输出路径和实验名，以及DARA显式记录correctness/format奖励通道。

本地Haotian与DARA源码对照中，DVAO优势实现和reward scorer的AST一致；actor的forward、log-prob、optimizer step和update_policy方法AST一致。两个日志均确认global seed0并使用XFormers backend。

| 观测 | 原始训练 | 复跑 |
|---|---:|---:|
| step1 Format reward | 0.00634765625 | 0.00634765625 |
| step1 Correctness reward | -2.443600893 | -2.443600893 |
| step1 grad norm | 37.385106862 | 37.373331487 |
| step3 Format reward | 0.01025390625 | 0.005859375 |
| step60 Format reward | 0.828125 | 0.0771484375 |
| step100 Format reward | 0.95263671875 | 0.0712890625 |

两次前两步的reward和平均response长度相同，每步平均prompt长度也一致。梯度统计从step1开始出现差异，生成与reward统计从step3开始分开，早于首次step10保存模型。现有记录将分叉定位在最早的策略更新阶段；分布式浮点计算及运行时非确定性是待验证的候选原因，现有日志尚不足以确定唯一触发点。

已完成的复跑训练、全部过程模型和BFCL逐case输出保留在原路径。过程比较使用DVAO seeds3/4/5；原始五seed最终比较维持既定结果。复跑过程评测的队列记录保存在BFCL研究目录的 `completed_runs/haotian_1p5b_dvao_s0_save10.json`，其原始模型评分目录继续保留。

原始训练目录：`/scratch.global/lian0190/RD-GDPO/results/figure1/dvao_seed0`。

复跑训练目录：`/scratch.global/lian0190/DARA/20260917/1p5b-dvao-g4-s0-two-save10`。

BFCL研究目录：`/scratch.global/lian0190/BFCL-v4-evaluation/20260917/dara_checkpoint_study`。
