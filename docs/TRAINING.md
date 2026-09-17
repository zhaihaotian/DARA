# 固定训练协议

先执行 `python scripts/fetch_data.py` 获取固定版本数据。同一模型规模内所有方法使用 `data/rlla_4k/train.parquet` 与 `test.parquet`，保持原 prompt、ground truth、顺序和 split。Haotian 加载为 3920 train / 80 validation。不要重新抽样、过滤或混入 BFCL 测试数据。

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
| training/data seeds | 0、1、2、4、5；三 seed 基线为 0、1、2 |
| rollout seed | 沿用 vLLM 引擎默认 0；不随 training seed 改变 |
| DARA weight cap | 5 |

GRPO 使用经过原 reward-KL 接线的总 reward；GDPO、DARA 和其他分通道方法读取原始通道 reward。保留此历史差异。动态 microbatch 的 token mean loss 仍除以名义累积次数，当前为 2；不替换成 Jay 的全局 token 缩放。FlashAttention 确定性设置与历史默认相同；不新增算法之间不同的确定性开关。

3B 已在四卡 A10040GB 跑完五 seeds。它使用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；launch.py 按 `--model-size 3b` 设置。不要为了跑 3B 更改 batch、response 长度、TP 或训练步数。模型路径与声明的规模必须一致。

## Group 消融的预算

| G | prompt batch | PPO mini（prompt 基数） | PPO micro（prompt 基数） | 回答/step |
|---:|---:|---:|---:|---:|
| 4 | 512 | 128 | 64 | 2048 |
| 8 | 256 | 64 | 32 | 2048 |
| 16 | 128 | 32 | 16 | 2048 |
| 32 | 64 | 16 | 8 | 2048 |

这是固定每步 rollout 数与 optimizer 更新预算的消融。G 变大时每步不同 prompt 数相应减少，这是实验操作变量的一部分。数据文件、数据 seed、学习率、训练步数、reward 定义和其余设置保持一致。不能保持 512 个 prompt 同时把 G 放大而称为相同预算实验。

## 三奖励

原两奖励是 correctness ∈[-3,3] 和 format ∈{0,1}。新增实验使用 `--rewards three`，额外加入原 GDPO/ToolRL scorer 中已有的 length reward：

`r_length = min(1, round(number_of_whitespace_words_in_think / 512, 2))`。

缺少 `<think>` 或 `</think>` 得 0；训练 scorer 取最后一个 `<think>` 后、紧接的 `</think>` 前的文本。最大 reward 1、最小 0，不启用 length schedule。这是鼓励思考长度到达 512 words 的奖励，并不是惩罚超长输出；512 是 whitespace words，不是 tokenizer tokens，也不修改 1024-token 生成上限。

GRPO 对三路原 reward 之和使用原有组内归一化与 KL 接线。GDPO 对三路分别做组内 z-score、相加后沿用 token whitening。DARA 对三路分别计算 π 与权重并双侧校准，再沿用同一个 token whitening；π_ref 取三路最大值。三路保持原尺度且不人为调权，不按结果搜索阈值。length 必须同时出现在训练总 reward、优势输入和日志中。

## 输出与中断

每个 run 保存 `launch.json`、`command.json`、展开的 `config.json`、`metrics.jsonl`、`exit.code` 和最终模型。最终模型只包含推理所需参数，不包含完整 optimizer / dataloader 恢复状态。不能将中断 run 称为已完成，也不能承诺从任意中间 step 精确续训；为完整 run 申请足够租期，中断后用新输出目录从相同 base 与 seed 重跑。

已完成的 3B G4 两奖励累计训练 step 耗时约 4–5 小时；申请至少约 6 小时并预留模型加载、验证、导出时间。G 消融及三奖励尚未完整实测耗时，尤其 length 会改变生成量，不能套用旧 ETA 当作保证。充分使用获配 GPU 时按四卡 slot 分配完整 run；CPU/RAM 可随集群调整，不能借此改训练参数。
