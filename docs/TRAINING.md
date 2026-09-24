# 固定训练协议

先执行 `python scripts/fetch_data.py` 获取固定版本数据。同一模型规模内所有方法使用 `data/rlla_4k/train.parquet` 与 `test.parquet`，保持原 prompt、ground truth、顺序和 split。Haotian 加载为 3920 train / 80 validation。

| 参数 | 固定值 |
|---|---|
| 模型 | Qwen2.5-1.5B-Instruct 或 Qwen2.5-3B-Instruct |
| 训练资源 | 每 run 单节点 4 × A100 40GB |
| 训练 / rollout | FSDP / vLLM TP=1，bf16 |
| 训练长度 | 完整 100 steps；训练前验证，之后每 10 steps 验证 |
| Checkpoint | step100 的 Hugging Face actor 模型与 tokenizer |
| 学习率 / PPO epochs | 1e-6 / 1 |
| PPO clip / grad clip | 0.2 / 1.0 |
| prompt / response 上限 | 2048 / 1024 tokens |
| rollout temperature / top_p / top_k | 1.0 / 1.0 / -1 |
| 每步回答数 | 2048 |
| 每个 optimizer minibatch 回答数 | 512；每训练 step 更新 4 次 |
| 动态 batching | True；每 GPU token budget 16384 |
| rollout memory utilization | 0.6 |
| actor param / grad / optimizer offload | False / False / False |
| ref param offload | True |
| gradient checkpointing / remove padding | True / True |
| entropy coefficient | 0.001 |
| actor use_kl_loss | False；原 reward KL coefficient 0.001 |
| training/data seeds | 历史最终比较为0、1、2、4、5；本轮过程比较按manifest指定，DVAO使用3、4、5 |
| rollout seed | 沿用 vLLM 引擎默认 0 |
| DARA weight cap | 5 |

GRPO 使用经过原 reward-KL 接线的总 reward；GDPO、DARA 和其他分通道方法读取原始通道 reward。动态 microbatch 的 token mean loss 除以名义累积次数，当前为 2。FlashAttention 使用同一套默认确定性设置。

3B 已在四卡 A10040GB 跑完五 seeds。它使用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；launch.py 按 `--model-size 3b` 设置。3B 沿用表中 batch、response 长度、TP 和训练步数，模型路径指定对应的 3B 权重。

当前补齐与过程比较每10steps保存，任务见 [CHECKPOINT_STUDY.md](CHECKPOINT_STUDY.md)。H100两卡执行四路逻辑分组的配置和验证见 [TWO_GPU_LOGICAL_FOUR.md](TWO_GPU_LOGICAL_FOUR.md)。两节点各两卡的资源布局、Ray/FSDP启动方式和当前接线状态见 [MULTINODE.md](MULTINODE.md)。

## Group 消融的预算

| G | prompt batch | PPO mini（prompt 基数） | PPO micro（prompt 基数） | 回答/step |
|---:|---:|---:|---:|---:|
| 4 | 512 | 128 | 64 | 2048 |
| 8 | 256 | 64 | 32 | 2048 |
| 16 | 128 | 32 | 16 | 2048 |
| 32 | 64 | 16 | 8 | 2048 |

这是固定每步 rollout 数与 optimizer 更新预算的消融。G 变大时每步不同 prompt 数相应减少，这是实验操作变量的一部分。数据文件、数据 seed、学习率、训练步数、reward 定义和其余设置保持一致。

## 三奖励

本分支的Length≥64二值实验使用 `--rewards three --length-min-words 64`，think区域达到64词得1，否则得0；具体八次训练与两档温度评测见 [HANDOFF_LENGTH64.md](HANDOFF_LENGTH64.md)。

原两奖励是 correctness ∈[-3,3] 和 format ∈{0,1}。新增实验使用 `--rewards three`，额外加入原 GDPO/ToolRL scorer 中已有的 length reward：

`r_length = min(1, round(number_of_whitespace_words_in_think / 512, 2))`。

缺少 `<think>` 或 `</think>` 得 0；训练 scorer 取最后一个 `<think>` 后、紧接的 `</think>` 前的文本。最大 reward 1、最小 0，length schedule 关闭。该奖励随思考文本长度增加，在约512个空白分隔词处达到上限；生成预算固定为1024 tokens。

GRPO 对三路原 reward 之和使用原有组内归一化与 KL 接线。GDPO 对三路分别做组内 z-score、相加后沿用 token whitening。DARA 对三路分别计算 π 与权重并双侧校准，再沿用同一个 token whitening；π_ref 取三路最大值。三路保持原尺度，length 同时参与训练总 reward、优势计算和日志统计。

## 输出与中断

每个 run 保存 `launch.json`、`command.json`、展开的 `config.json`、`metrics.jsonl`、`exit.code` 和最终模型。最终模型保存为 Hugging Face 权重与 tokenizer。为完整 run 申请足够租期；训练中断后，用新输出目录从相同 base 与 seed 重跑。

需要保留训练过程模型时，启动命令加入 `--save-freq 10`，在steps10/20/…/100分别保存到 `actor/global_step_<step>`。每份包含模型权重与tokenizer，供相应训练阶段的推理和评测使用；验证频率仍为每10步。默认 `--save-freq 100` 保存最终模型。

GRPO、GDPO、DARA每step在 `metrics.jsonl` 中记录各奖励通道的π、权重和活跃group数，两奖励包含correctness/format，三奖励再包含length。日志字段与导出训练曲线的命令见 [HANDOFF.md](HANDOFF.md#训练-dynamics)。

已完成的 3B G4 两奖励累计训练 step 耗时约 4–5 小时；申请至少约 6 小时并预留模型加载、验证、导出时间。G 消融与三奖励的耗时在首个完整 run 后更新，length 会影响生成量。获配A100按四卡一组、H100按两卡一组分配完整run，CPU/RAM按集群资源配置。H100四路逻辑分组的实际耗时在首个完整run后更新。
