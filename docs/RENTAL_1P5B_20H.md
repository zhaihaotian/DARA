# 1.5B 八卡 A100 80GB SXM4 租机实验清单

机器为8×A100 80GB SXM4、租期20小时。目标为两卡保留四路逻辑分组、同时运行四个训练实验。独立的执行agent交接文档为 [RENTAL_A100_80G_SXM4_AGENT.md](RENTAL_A100_80G_SXM4_AGENT.md)，包含两卡实现与验证、GPU持续补位、环境、启动、评测和交付要求。任务清单位于 [rental_1p5b_20h.json](../configs/rental_1p5b_20h.json)。服务器获配后登记任务归属并移交集群中的对应待运行任务。

## 固定设置

全部使用Haotian infra、Qwen2.5-1.5B-Instruct、G4、correctness+format两奖励、512个prompt/step、2048条回答/step、学习率1e-6、100steps。每10steps保存HF模型并验证，保留steps10/20/…/100。原数据、reward、动态microbatch、loss累积、KL、entropy与rollout seed设置沿用 [TRAINING.md](TRAINING.md)。训练环境安装使用 [HANDOFF.md](HANDOFF.md) 的 `rdgdpo` 环境流程。

## 过程比较的种子选择

GRPO、GDPO、DARA使用原始五seed实验中训练Format Reward首次达到0.8的step，按从快到慢排序；未达到阈值的seed排在末尾，并列按seed编号升序排列。去掉排序首尾各一个，保留中间三个。最终比较继续汇总完整seed结果，过程比较记录实际使用的seed。

| 方法 | seed0 / 1 / 2 / 4 / 5 的首次达标step | 去掉最快seed | 去掉最慢seed | 过程重跑seed |
|---|---|---|---|---|
| GRPO | 37 / 43 / 34 / 31 / 31 | 4（与5并列，按编号处理） | 1 | 0、2、5 |
| GDPO | 18 / 19 / 75 / 16 / 65 | 4 | 2 | 0、1、5 |
| DARA | 14 / 16 / 15 / 13 / 29 | 4 | 5 | 0、1、2 |

DVAO使用seed3、4、5做过程比较，新增seed3过程训练，同时补齐seed4、5的最终训练并保存全部过程模型。GD²PO-Hard使用seed0、4、5；原始seed0首次达到Format0.8为step26，重跑该seed并补齐seed4、5。

DVAO seed3完成steps10/20/…/100的BFCL V4过程评测，同时交付step100最终评测结果。本轮最终结果表覆盖全部15次训练，DVAO汇总seed3/4/5；历史五seed最终比较沿用原始seed0/1/2及新增4/5。已完成的DVAO seed0重跑日志、模型和评测结果单独保留，复跑对照见 [DVAO_SEED0_REPLAY.md](DVAO_SEED0_REPLAY.md)。

## 需要执行的15次训练

| 顺序 | 方法 | Seed | 目的 | Run ID |
|---:|---|---:|---|---|
| 1 | DVAO | 4 | 补齐最终比较，保存过程模型 | `haotian_1p5b_dvao_s4_save10` |
| 2 | GD²PO-Hard | 4 | 补齐最终比较，保存过程模型 | `haotian_1p5b_gd2po_hard_s4_save10` |
| 3 | DVAO | 5 | 补齐最终比较，保存过程模型 | `haotian_1p5b_dvao_s5_save10` |
| 4 | GD²PO-Hard | 5 | 补齐最终比较，保存过程模型 | `haotian_1p5b_gd2po_hard_s5_save10` |
| 5 | GRPO | 0 | 过程checkpoint比较 | `haotian_1p5b_grpo_s0_save10` |
| 6 | GDPO | 0 | 过程checkpoint比较 | `haotian_1p5b_gdpo_s0_save10` |
| 7 | DARA | 0 | 过程checkpoint比较 | `haotian_1p5b_dara_s0_save10` |
| 8 | GRPO | 2 | 过程checkpoint比较 | `haotian_1p5b_grpo_s2_save10` |
| 9 | GDPO | 1 | 过程checkpoint比较 | `haotian_1p5b_gdpo_s1_save10` |
| 10 | DARA | 1 | 过程checkpoint比较 | `haotian_1p5b_dara_s1_save10` |
| 11 | GRPO | 5 | 过程checkpoint比较 | `haotian_1p5b_grpo_s5_save10` |
| 12 | GDPO | 5 | 过程checkpoint比较 | `haotian_1p5b_gdpo_s5_save10` |
| 13 | DARA | 2 | 过程checkpoint比较 | `haotian_1p5b_dara_s2_save10` |
| 14 | GD²PO-Hard | 0 | 历史收敛seed的过程checkpoint重跑 | `haotian_1p5b_gd2po_hard_s0_save10` |
| 15 | DVAO | 3 | 过程checkpoint训练、逐step评测及step100最终评测 | `haotian_1p5b_dvao_s3_save10` |

两卡四路对齐通过后，先并行执行前四个补种子任务，再动态补入其余十一个过程训练，四条通道约分配4/4/4/3次。准备期间采用当前四卡两路执行。完成15次训练后，五个方法各有三个seed的完整过程checkpoint，同时补齐1.5B最终比较。

## 时间与输出

历史四卡40GB的1.5B完整训练约2.3–2.6小时，最近每10steps保存的DVAO训练端到端耗时2小时28分钟。新机器的两卡完整run耗时需实测；四路并行的训练预算约为四个两卡run的耗时，环境准备、GPU验证、数据回传和BFCL评测另计。按实测整机吞吐选择布局，按剩余租期安排完整run，空出的GPU立即接训练或过程评测。

每个run保存独立输出目录、展开配置、完整训练日志和十份HF checkpoint。GRPO、GDPO、DARA记录两路reward、π、权重和活跃group数量。用tmux维持持续派发任务的控制器，记录每项exit.code并即时补位。

按现有DVAO训练导出的文件实测，一份1.5B HF checkpoint约6.64GiB，15个run的150份模型约996GiB。机器建议配置2TB可持久保存的空间，另留环境、日志和评测输出空间。

## 评测与执行交接

全部15次训练都要完成steps10/20/…/100的BFCL V4评测，共150份checkpoint结果。第15项DVAO seed3同时交付step100最终评测结果。所有模型使用14类3301cases及相同推理与评分参数，报告Non-Live、Live、Multi-Turn、Average的Accuracy和Format，按方法和step汇总三个seed的均值与样本标准差。

训练启动、两卡改造与对照、评测命令、CPU表格导出以及每分钟GPU补位要求统一放在独立的 [agent交接文档](RENTAL_A100_80G_SXM4_AGENT.md)。**任何时候最大化利用所有已租GPU**：可开完整训练时立即开训，训练尾段或剩余租期较短时立即接checkpoint评测。评分完成后的四份CSV为 `checkpoint_per_model.csv`、`checkpoint_method_results.csv`、`process_final_per_model.csv`、`process_final_method_results.csv`，分别交付过程数据和本轮最终结果。评测定义见 [EVALUATION.md](EVALUATION.md)。
