# 协作者与执行 agent 交接

本次目标是用一个干净仓库继续DARA研究。代码采用Haotian infra，所有现有结果和接下来的比较均按本仓库固定协议执行。先读 [ENVIRONMENT.md](ENVIRONMENT.md)、[TRAINING.md](TRAINING.md)、[EVALUATION.md](EVALUATION.md) 和 [STATUS.md](STATUS.md)，运行测试和 `python scripts/fetch_data.py` 后再领取manifest里的run。不要把环境升级、训练infra修正、reward parser变更或评测口径探索混在这些实验里。

## 需求与交付位置

| 用户要求 | 本仓库对应交付 |
|---|---|
| 独立repo，方法名DARA | 本仓库；DARA仅使用双侧校准 |
| 保留自己的infra和已有算法 | vendor/verl、training/launch.py、ALGORITHMS.md |
| 环境可让别人创建 | 四份环境快照和依赖锁定、安装脚本 |
| 训练数据不变 | scripts/fetch_data.py获取逐字节相同的3920/80原始parquet |
| 明确V3/V4及average/format口径 | EVALUATION.md和evaluation/protocol.py |
| 交代五seed/三seed和未收敛结果 | STATUS.md、完整表、逐seed CSV、筛选诊断表 |
| 说明SAW/RVPO/DVAO/Hard现状 | STATUS.md |
| Group大小消融 | configs/group_ablation.json |
| 真正训练三reward并评测长度 | configs/three_rewards.json；WITHLENGTH与分通道优势接线；length诊断 |
| 自行画training dynamics/density | training_dynamics.csv、42份原始指标JSONL |
| 可重复执行和核对 | scripts/test.sh、VALIDATION.md、逐run命令与配置记录 |

## 实验 A：Group size

使用Qwen2.5-1.5B-Instruct；GRPO、GDPO、DARA；G=4/8/16/32；每格seeds0/1/2/4/5。共60个run，其中G4的15个已经完成、标为`reuse_complete`；新增45个标为`planned`。仅改变G及维持每步2048回答所必需的prompt/minibatch/microbatch基数。详细值见TRAINING.md。

manifest是任务清单，不是集群状态数据库。执行时将领取的run ID记录到自己的调度日志，避免多人同时做同一个run。每run单节点四卡A10040GB；有8卡就运行两个完整run，有16卡就四个，继承allocation分配的GPU可见范围。跨方法先覆盖seeds0/1/2，再做4/5。保持获配资源利用率，但不能以缩步数、缩回答长度、提前选checkpoint替代完整实验。

```bash
conda activate rdgdpo
python training/run_manifest.py --manifest configs/group_ablation.json \
  --id 1p5b-gdpo-g8-s0-two --model-root /models --output-root /experiments/dara
```

训练完成后确认恰有steps1–100、validation steps0/10/…/100、final模型可加载，再按相同V3/V4协议评测。所有run的曲线用同一横轴和汇总方式。密度π只在DARA历史日志中直接记录；GRPO/GDPO的平均reward不能唯一恢复prompt-group活动比例，缺失密度不得填0或推断成准确π。若后续要为所有方法记录π，需要明确另行增加只读统计，不能更改优势。

## 实验 B：correctness + format + length

使用Qwen2.5-1.5B-Instruct和Qwen2.5-3B-Instruct；GRPO、GDPO、DARA；每格五seeds0/1/2/4/5；G固定4；100steps，共30个新增run。两奖励对照直接使用已有同规模、同seed结果。除增加length reward及其通道外保持全部参数和训练数据不变。

```bash
python training/run_manifest.py --manifest configs/three_rewards.json \
  --id 1p5b-dara-g4-s0-three --model-root /models --output-root /experiments/dara
```

length定义固定为原scorer的think whitespace words/512、round2、cap1；它鼓励长度而不是压缩长度。正式run必须同时看到`critic/length_score/mean`和三奖励配置；DARA另有`dara/pi_length`、`dara/w_length`。不能只给旧checkpoint补一个Length表然后称为三奖励实验。

最终模型仍按V3两组AST与V4三组评测，并在同一套原始输出上统计Format*、Length Reward、Length≥512和think words；分别报告，保持现有accuracy计算。比较中要能看到长度是否达标，以及正确率和格式是否维持。若长度没有学到，保留失败结果并报告；不偷偷调整512阈值、解码预算或数据。

## 执行和交回

用tmux等持久终端运行；训练每个输出目录只能归属一个run，评测每个目录只能归属一个checkpoint和一个版本。训练中断不能从当前仅保存最终模型的方案精确恢复optimizer，应按TRAINING.md处理。评测可直接重复命令补齐缺失case。

交回每个run的配置、命令、完整metrics.jsonl、100step最终模型位置和可加载记录；每次评测交回原始raw JSONL、官方score、逐case诊断和summary.json。汇总表保留所有seed，采用mean±sample SD；另有筛选分析时必须标明筛选依据和样本数。让汇总表能追溯到原始case和模型，不只交回图片。

不要为这次交接自动扩大到3B的四个附加基线，不恢复旧单侧比较，不再训练MALT。当前明确新增任务只有上面45+30个run；四个附加基线的代码和已有1.5B三seed结果保留供后续使用。所有未经授权改变的训练和评测行为视为冻结。遇到确定的执行错误修局部根因并记录，不能通过更改任务定义让run“通过”。
