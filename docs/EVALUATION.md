# BFCL evaluation protocol

## 模型与推理

最终评测使用每个训练 run 的 step100 模型，并加入对应规模的 base 模型。1.5B过程比较还对指定run的steps10/20/…/100逐个执行BFCL V4评测。每个 checkpoint 对每个 case 生成一条 trajectory。所有方法使用相同的 ToolRL prompt、输出解析器、数据子集和解码参数。

| 项目 | V3 | V4 |
|---|---|---|
| Evaluator | bfcl-eval 2025.8.6.2 | gorilla commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` |
| 评测子集 | Non-Live AST + Live AST，2501 cases | Non-Live AST + Live AST + Multi-Turn，3301 cases |
| Prompt / 解析 | ToolRL RLLA handler | ToolRL RLLA handler，适配 V4 API |
| 接口 | completion；is_fc_model=False | completion；is_fc_model=False |
| temperature / top_p | 0 / 1 | 0.6 / 0.95 |
| 最大新 tokens / context | 4096 / 32768 | 8192 / 32768 |
| seed / top_k / repetition penalty | 0 / -1 / 1 | 0 / -1 / 1 |
| vLLM 服务 | 0.11.0，TP1，bf16，memory0.85，enforce eager | 同左 |
| 服务 batching | max_num_seqs64 / max_num_batched_tokens8192；48 concurrent cases | 同左 |

Prompt 和 parser 位于 `evaluation/v3/toolrl_adapter.py` 与 `evaluation/v4/toolrl_adapter.py`。`adapter.py` 为每次请求记录 single/multi-turn 类型；V4 在每条对话开始时初始化该 case 的工具状态。Multi-Turn 的工具执行与对话推进由 BFCL driver 完成。

## Accuracy

每类别 accuracy 使用 BFCL 官方 checker。AST 评测检查解析后的函数名、参数与可接受答案；Multi-Turn 评测检查工具执行状态和对话结果。

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
| Multi-Turn Base | — | multi_turn_base | 200 |
| Multi-Turn Missing Function | — | multi_turn_miss_func | 200 |
| Multi-Turn Missing Parameter | — | multi_turn_miss_param | 200 |
| Multi-Turn Long Context | — | multi_turn_long_context | 200 |

Non-Live 共1150题，按以下分层平均计算：

$$
A_{\mathrm{simple}}=\frac{A_{\mathrm{Python}}+A_{\mathrm{Java}}+A_{\mathrm{JavaScript}}}{3},
$$

$$
A_{\mathrm{nonlive}}=\frac{A_{\mathrm{simple}}+A_{\mathrm{multiple}}+A_{\mathrm{parallel}}+A_{\mathrm{parallel\_multiple}}}{4}.
$$

Live 共1351题，按样本数加权：

$$
A_{\mathrm{live}}=\frac{\sum_{c\in\mathrm{Live}}N_c A_c}{1351}.
$$

Multi-Turn 的四类各200题，取四类准确率的算术平均。V4 的 **Average** 为三组准确率的算术平均：

$$
\mathrm{Average}=\frac{A_{\mathrm{nonlive}}+A_{\mathrm{live}}+A_{\mathrm{multiturn}}}{3}.
$$

V3 的结果表报告 Non-Live AST 和 Live AST。V4 报告这两组、Multi-Turn 和 Average。

## Format 与 Average Format

Format 衡量 assistant 输出是否符合规定的标签结构。先按 Qwen assistant marker 规则提取正文并去掉两端空白，再检查以下三种格式；符合任意一种得1，否则得0。

| 格式 | 必需结构 |
|---|---|
| 文本回复 | `<think>...</think>\n<response>...</response>` |
| 工具调用 | `<think>...</think>\n<tool_call>\n...\n</tool_call>` |
| 工具调用与文本回复 | `<think>...</think>\n<tool_call>\n...\n</tool_call>\n<response>...</response>` |

检查覆盖整条输出的标签顺序、闭合、换行，以及 tool_call / response 标签各出现一次的要求。实现直接调用 `rlla.py::customize_format_reward_func` 的三种分支。

对于 case i，记其中 assistant 输出的条数为 m_i、每条输出的格式得分为 f_ij，则：

$$
F_i=\frac{1}{m_i}\sum_{j=1}^{m_i}f_{ij}.
$$

空输出 case 得0。类别 Format 为该类别所有 case 的 F_i 的算术平均。随后按 Accuracy 相同的 Non-Live、Live、Multi-Turn 权重汇总。V4 的 **Average Format** 为：

$$
\mathrm{Average\ Format}=\frac{F_{\mathrm{nonlive}}+F_{\mathrm{live}}+F_{\mathrm{multiturn}}}{3}.
$$

`evaluation/diagnostics.py` 计算逐 case 指标，`evaluation/protocol.py` 计算各组分数。Accuracy 和 Format 在结果表中以百分数报告。

## Seeds 与汇总

每个 checkpoint 先独立完成评测、计算组分数，再对 training seeds 报告 mean ± sample standard deviation（ddof=1）。历史结果中GRPO、GDPO、DARA在1.5B与3B均有seeds0/1/2/4/5；1.5B的额外基线有seeds0/1/2，本轮DVAO与GD²PO-Hard补齐4/5。Base使用一份固定协议输出。推理seed统一为0。

## 过程checkpoint与本轮最终结果

本轮1.5B过程比较包含GRPO seeds0/2/5、GDPO seeds0/1/5、DARA seeds0/1/2、DVAO seeds3/4/5、GD²PO-Hard seeds0/4/5。每个run评测steps10/20/…/100，共15个run、150份BFCL V4结果，每份覆盖14类3301cases。按方法和训练step分别汇总三个seed的八项Accuracy/Format指标，绘制它们随训练step变化的曲线。种子选择依据见 [RENTAL_1P5B_20H.md](RENTAL_1P5B_20H.md)。

本地3B DVAO seed0与GD²PO-Hard seed0的既有steps10/20/…/100也按相同协议评测，step100直接复用已有结果。3B过程记录目前每个方法为一个seed，按模型尺寸和训练step单独列出。Haotian单侧RD-GDPO补评1.5B五seed与3B四seed的step100模型，用于与双侧的V4结果对照。

过程run各自的step100同时生成最终评测结果，包含租机15个1.5B run（含第15项DVAO seed3）和本地两个3B seed0 run。step100的推理与评分执行一次，该结果同时写入过程数据和本轮最终结果表。使用 `evaluation/checkpoint_results.py` 汇总后，过程数据写入 `checkpoint_per_model.csv`、`checkpoint_method_results.csv`，本轮最终结果写入 `process_final_per_model.csv`、`process_final_method_results.csv`。逐模型表保留run ID、模型规模、方法、seed、step与summary路径；汇总表保留完成seed数、目标seed数、均值和样本标准差。命令见 [租机agent评测交接](RENTAL_A100_40G_AGENT.md#bfcl-v4协议与交付)。

## 三奖励实验的长度指标

Length≥64实验使用 `evaluation/run.py --version v4 --temperature 0.6 --top-p 1 --length-min-words 64` 与 `--temperature 1 --top-p 1 --length-min-words 64` 两档。配置保存到inference.json，评分自动读取阈值；`groups.length`为二值Length Reward，`groups.length_ge_64`为达标比例，输出→case→category→group的平均顺序与下述协议一致。运行和汇总命令见 [HANDOFF_LENGTH64.md](HANDOFF_LENGTH64.md)。

在相同 BFCL 输出上，提取最后一个 `<think>` 后、紧接的 `</think>` 前的文本，以空白分隔计算词数 n。Length Reward 为：

$$
R_{\mathrm{length}}=\min\left(1,\operatorname{round}\left(\frac{n}{512},2\right)\right).
$$

缺少 think 标记得0。`Length≥512` 在 think 标记存在且 n≥512 时得1，否则得0；同时记录平均 think words。

长度指标沿用输出→case→category→group 的平均顺序。结果表分别报告 Accuracy、Format、Length Reward 和 Length≥512；Length Reward 保留0–1尺度，Length≥512以百分数报告。

## 运行评测

按 [ENVIRONMENT.md](ENVIRONMENT.md) 创建环境。每个 checkpoint 用一张 GPU 推理，四张卡可以并行评测四个 checkpoint，各自使用独立的输出目录与端口。

```bash
conda activate dara-bfcl-v4
CUDA_VISIBLE_DEVICES=0 python evaluation/run.py --version v4 \
  --model /experiments/1p5b-dara-g8-s0-two/actor/global_step_100 \
  --server-python /path/to/envs/dara-inference/bin/python \
  --port 8000 --output outputs/eval-v4/1p5b-dara-g8-s0-two
```

V3 使用 `dara-bfcl-v3` 环境、`--version v3` 和对应输出目录。用 tmux 保持任务运行，并将终端输出保存到日志。

每条推理结果立即写入 `raw/<category>.jsonl`。重复同一命令会补齐缺失 case。服务中断等基础设施错误通过补跑处理；超过既定 context 的 case 计入模型失败。类别样本齐全后执行官方评分。

对已保存的结果重新评分使用 CPU：

```bash
python evaluation/score.py --version v4 --output outputs/eval-v4/1p5b-dara-g8-s0-two
```

产物包括官方 `result/`、`score/`、逐 case 的 `diagnostics_per_case.csv` 和各组指标的 `summary.json`。完整的协作者运行流程见 [HANDOFF.md](HANDOFF.md)，论文文字见 [PAPER_EXPERIMENTS.md](PAPER_EXPERIMENTS.md)。
