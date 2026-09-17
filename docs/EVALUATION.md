# BFCL evaluation protocol

## 模型与推理

所有训练方法只评测 **final step100**，并加入同一模型规模的 base。每个训练 seed 的 checkpoint 评测一次；training seed 数量与 inference 重采样次数不同。V3 与 V4 使用各自固定的协议，并对所有方法统一应用，不按模型改变 prompt、解析器、生成预算或子集。

| 项目 | V3 | V4 |
|---|---|---|
| 官方 evaluator | bfcl-eval 2025.8.6.2 | gorilla commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` |
| 当前报告范围 | Non-Live AST + Live AST，2501 cases | Non-Live AST + Live AST + Multi-Turn，3301 cases |
| Prompt / 输出解析 | 原 ToolRL RLLA handler | 同一 ToolRL 格式、适配 V4 API |
| 接口 | completion；is_fc_model=False | completion；is_fc_model=False |
| temperature / top_p | 0 / 1 | 0.6 / 0.95 |
| 最大新 tokens / context | 4096 / 32768 | 8192 / 32768 |
| seed / top_k / repetition penalty | 0 / -1 / 1 | 0 / -1 / 1 |
| 每个 case | 一条 trajectory | 一条 trajectory；多轮由官方 driver 驱动 |
| vLLM 服务 | 0.11.0，TP1，bf16，memory0.85，enforce eager | 同左 |
| 服务 batching | max_num_seqs64 / max_num_batched_tokens8192；48 concurrent cases | 同左 |

ToolRL 的 system prompt 与 parser 实体保存在 `evaluation/v3/toolrl_adapter.py`、`evaluation/v4/toolrl_adapter.py`。`adapter.py` 把 single/multi-turn 标识保存在每个请求自己的数据里；V4 每次对话重建该 case 的工具状态，以免并发或重试污染下一次执行。不要改成模型自带 function-calling 模板后仍引用这些参考数值。

## 子集及分母

| 子集 | V3 category | V4 category | Cases |
|---|---|---|---:|
| Simple Python | simple | simple_python | 400 |
| Simple Java | java | simple_java | 100 |
| Simple JavaScript | javascript | simple_javascript | 50 |
| Multiple | multiple | multiple | 200 |
| Parallel | parallel | parallel | 200 |
| Parallel Multiple | parallel_multiple | parallel_multiple | 200 |
| Live Simple | live_simple | live_simple | 258 |
| Live Multiple | live_multiple | live_multiple | 1053 |
| Live Parallel | live_parallel | live_parallel | 16 |
| Live Parallel Multiple | live_parallel_multiple | live_parallel_multiple | 24 |
| Multi-Turn Base | 当前V3主表不报告 | multi_turn_base | 200 |
| Multi-Turn Missing Function | 当前V3主表不报告 | multi_turn_miss_func | 200 |
| Multi-Turn Missing Parameter | 当前V3主表不报告 | multi_turn_miss_param | 200 |
| Multi-Turn Long Context | 当前V3主表不报告 | multi_turn_long_context | 200 |

每类别 accuracy 由 pinned BFCL 官方 checker 给出。AST 检查解析后的函数名、参数与可接受答案；Multi-Turn 使用官方执行状态和对话检查器。没有 LLM judge，也不以 format reward 替代任务正确率。官方容许等价调用时，输出不符合训练时的外层标签仍可能 AST 正确。

`Simple = mean(Python, Java, JavaScript)`；`Non-Live AST = mean(Simple, Multiple, Parallel, Parallel Multiple)`。因此 Non-Live 是分层 macro mean，不是把1150题全部直接合并；Java与JavaScript在Simple内部各占1/3。

`Live AST = sum(correct_count of four Live categories) / 1351`，即 example-weighted micro mean。`Multi-Turn = mean(Base, Missing Function, Missing Parameter, Long Context)`。

**V4 Avg = (Non-Live AST + Live AST + Multi-Turn)/3。** 这是当前三组协议的平均，不是包含 Agentic、Web Search、Memory、Relevance/Irrelevance 等全部项目的 BFCL V4 Overall。V3 当前主表只报告两组 AST；不会把它的平均误写成完整 V3 leaderboard Overall。历史探索中存在更广的 V3 评测，其结果不与这两组口径混算。

跨 training seeds 先分别算出每个 checkpoint 的组分数，再报告算术平均与 sample standard deviation（ddof=1）。不把不同 seed 的所有 cases 直接池化来计算“跨 seed 方差”。Base只有一份输出，不写成五seed均值。

## Format* 与 Average Format*

Format* 是我们的 RLLA 结构诊断，不是 BFCL 官方 accuracy，也没有宣称复现了 GDPO 论文未完整公开的 Correct Format 数字。

先按训练 scorer 的 Qwen assistant marker 规则去掉包装及两端空白。每次 assistant emission 若符合以下任一结构，就得1，否则0：`<think>…</think>` 后为 `<response>…</response>`；或带必要换行和唯一闭合标签的 `<tool_call>`；或按顺序同时包含工具调用与 response。具体锚定正则、换行和标签计数要求直接调用仓库 `rlla.py::customize_format_reward_func`。纯 think 不能通过此评测诊断。

BFCL 没有训练集那样的 ground-truth 标签选择器，所以这里测试三种允许分支的并集；这和训练时依据答案选择某一分支的 format reward 不完全相同。Format* 检查结构，不判断函数/参数是否正确，不要求“正确且格式正确”才计1。

多轮或一次 case 多次 emission：先对该 case 所有 assistant emission 的0/1分数求平均，再对 case 等权求类别平均。空输出 case 为0；较长对话不会因 emission 更多而在类别级获得更大权重。之后使用与 accuracy 完全相同的 Non-Live、Live、Multi-Turn 权重。

**V4 Avg Format* = (Non-Live Format* + Live Format* + Multi-Turn Format*)/3。** `evaluation/diagnostics.py` 输出逐 case 记录，`evaluation/protocol.py` 是唯一聚合实现。CSV 内 `groups` 数值为0–1，论文表乘100；长度 reward 仍是0–1。

## 新三奖励实验的长度评测

使用同一批 BFCL 保存的原始输出，调用与训练相同的 length scorer：think 区域 whitespace words /512、四舍五入两位、上限1；缺少think标记为0。报告 `Avg Length Reward`（0–1）、`Length≥512` 比例及平均 think words。Length≥512 是明确长度达标判定；不能以 rounded reward=1 代替它，因为510words就可能四舍五入到1。

多轮使用 emission→case→category→group 的同一平均顺序。`Length≥512` 的三组平均是协议加权的比例，不是整个3301题直接池化的比例。保留 Accuracy 和 Format*；不把三者临时加成新的官方 Overall。这样可以观察新三奖励训练是否学到了长度要求，以及任务正确率和格式表现是否保留。

此前 GD²PO/API-Bank 探索使用作者 evaluator，提取第一个闭合 think 区域；本仓库新三奖励以**训练 scorer**的提取规则为准，针对多个think标签的异常文本两者可能不同。正常单think输出一致。已有两奖励模型的 length 后处理不能当作“三奖励训练结果”。

## 执行

先按 [环境说明](ENVIRONMENT.md) 创建环境。推理用一个 GPU 加载一个 checkpoint；四张卡可并行评测四个独立 checkpoint，每个使用不同输出目录与端口。训练的每 run 四卡要求不意味着评测也要 TP4。

```bash
conda activate dara-bfcl-v4
CUDA_VISIBLE_DEVICES=0 python evaluation/run.py --version v4 \
  --model /experiments/1p5b-dara-g8-s0-two/actor/global_step_100 \
  --server-python /path/to/envs/dara-inference/bin/python \
  --port 8000 --output outputs/eval-v4/1p5b-dara-g8-s0-two
```

V3 换为 `dara-bfcl-v3` 环境、`--version v3` 与单独输出目录。checkpoint 和 base 使用相同入口。使用 tmux 承载命令并将终端输出重定向到日志；断开终端不应杀掉任务。

每条推理结果立即写到 `raw/<category>.jsonl`，记录多轮完整结果和官方 metadata；重跑同一命令仅补缺失 case。基础设施错误不充当模型答错，应补齐后评分；超出既定 context 的输出保留为模型失败，不丢弃。官方评分必须在所有类别 case 数完整时执行。

重新 judge 已保存结果只需 CPU：

```bash
python evaluation/score.py --version v4 --output outputs/eval-v4/1p5b-dara-g8-s0-two
```

输出包含官方 `result/`、`score/`、`diagnostics_per_case.csv` 和 `summary.json`。相同输出的 AST/执行评分是确定性的；改变聚合口径不需要重新生成。但更换 BFCL 数据版本、prompt 或 decoding 会产生新的实验，不能从旧 rollout 直接宣称复现新设置。
