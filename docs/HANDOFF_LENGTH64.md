# Length≥64 二值三奖励实验交接

本轮使用 Haotian infra、Qwen2.5-1.5B-Instruct、G=4，对 GRPO、GDPO、DARA 双侧、DARA 单侧各训练 seeds 1、2。总计8次100-step训练；每个step100模型做两档温度的BFCL V4 final evaluation，总计16次评测。可执行清单为 [length_ge64_binary.json](../configs/length_ge64_binary.json)。

## 研究问题与奖励

本轮检验 length 奖励的离散化能否让 DARA 的密度校准更有效。原length为 `min(1, round(n/512, 2))`，本轮改成 `r_length = 1 if n >= 64 else 0`，其中n为think区域的空白分隔词数。提取规则沿用训练scorer：取最后一个`<think>`后、紧接的`</think>`前的内容；缺少任一标记得0。阈值按词数计算，正好64词得1。Correctness范围[-3,3]，Format范围{0,1}，Length范围{0,1}。

π衡量一个reward通道中具有非零组内优势的prompt group比例。连续length可能让许多group保持组内差异；二值length只有在同组回答跨越64词阈值时才产生组内优势。因此需结合π_length、通道权重、活跃group数、三路reward的训练动态，以及final下游指标来判断效果。实验假设是二值化能改变length通道的活跃密度，并改善多奖励协调；实际密度与效果由这些记录检验。

## 获取分支与环境

```bash
git clone --branch experiment/length-ge64-binary git@github.com:zhaihaotian/DARA.git DARA-length64
cd DARA-length64
bash environments/install_training.sh rdgdpo
conda activate rdgdpo
python scripts/fetch_data.py
export DARA_MODELS=/path/to/dara-models
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir "$DARA_MODELS/Qwen2.5-1.5B-Instruct"
bash scripts/test.sh
```

已有训练环境时使用现成rdgdpo，按 [ENVIRONMENT.md](ENVIRONMENT.md) 核对依赖。训练栈为Python3.10、Torch2.4.0+cu121、vLLM0.6.3、Ray2.10.0、Transformers4.47.1、FlashAttention2.6.3；依赖锁定和安装脚本位于environments/。

## 训练任务与参数

| 方法 | CLI method | Seeds | Run ID |
|---|---|---|---|
| GRPO | grpo | 1、2 | `1p5b-grpo-g4-s{1,2}-three-len64` |
| GDPO | gdpo | 1、2 | `1p5b-gdpo-g4-s{1,2}-three-len64` |
| DARA 双侧 | dara | 1、2 | `1p5b-dara-g4-s{1,2}-three-len64` |
| DARA 单侧 | rdgdpo_positive | 1、2 | `1p5b-rdgdpo_positive-g4-s{1,2}-three-len64` |

每run使用四张物理GPU、四个训练rank，100steps。固定同一份RLLA train/validation数据、prompt batch512、G4、2048回答/step、每step四次optimizer更新、学习率1e-6、PPO epoch1、动态batch token budget16384/GPU、prompt/response上限2048/1024、训练T=1/top_p=1、vLLM seed0、entropy0.001、原reward-KL系数0.001、原Haotian loss累计和whitening。DARA weight cap5。训练前验证，之后每10step验证；保存step100模型。其余固定参数见 [TRAINING.md](TRAINING.md)。

GRPO将三路reward相加后使用原组内归一化。GDPO逐通道标准化后相加。DARA逐通道计算密度与权重：`dara`执行双侧校准，`rdgdpo_positive`执行单侧校准。四方法共用相同binary length scorer，训练与验证均通过`LENGTH_MIN_WORDS=64`生效；launcher会记录该设置并传递给Ray worker。

在已获配四卡的节点使用tmux启动：

```bash
tmux new -s dara-length64
conda activate rdgdpo
cd /path/to/DARA-length64
export DARA_MODELS=/path/to/dara-models
export LENGTH64_RUN=1p5b-dara-g4-s1-three-len64
mkdir -p outputs/length64/logs
python training/run_manifest.py --manifest configs/length_ge64_binary.json \
  --id "$LENGTH64_RUN" --model-root "$DARA_MODELS" --output-root outputs/length64 \
  > "outputs/length64/logs/$LENGTH64_RUN.log" 2>&1
```

启动前给同一命令加入`--dry-run`可检查实际参数。最终模型路径为 `outputs/length64/<run-id>/actor/global_step_100`；单个run的直接入口等价于 `training/launch.py --method dara --seed 1 --model-size 1.5b --model /path/to/model --rewards three --length-min-words 64 --group-size 4 --save-freq 100 --output outputs/length64/<run-id>`。

任何时候都最大化利用已租GPU。八卡机器同时运行两个四卡训练，GPU0–3和4–7各执行一个run，结束立即领取清单中下一个run。用tmux维护任务；有可用GPU和可执行任务时立即补位。新训练的GPU任务启动后再做CPU曲线导出、打包和提交。训练队列完成后，每个BFCL评测使用一张GPU，八卡可同时执行八份评测；同一模型的两档温度也可各用一张卡并行执行。

## 两档温度的BFCL V4 final evaluation

| Profile / 输出目录 | Temperature | top_p | Length评分 |
|---|---:|---:|---|
| `t0p6-p1` | 0.6 | 1.0 | think词数≥64得1 |
| `t1p0-p1` | 1.0 | 1.0 | think词数≥64得1 |

两档均评测step100、14个类别3301题，覆盖Non-Live AST、Live AST和Multi-Turn；每题一条trajectory，固定inference seed0、top_k=-1、repetition penalty1、max_new_tokens8192、context32768、TP1、bf16、48 concurrent cases。官方checker、ToolRL prompt/parser及类别权重使用 [EVALUATION.md](EVALUATION.md) 中的V4定义。本轮两档top_p都为1，比较变量为temperature。

```bash
bash environments/install_inference.sh dara-inference
bash environments/install_evaluation.sh v4 dara-bfcl-v4
conda activate dara-inference
export DARA_SERVER_PYTHON="$(command -v python)"
conda activate dara-bfcl-v4
export LENGTH64_RUN=1p5b-dara-g4-s1-three-len64

CUDA_VISIBLE_DEVICES=0 python evaluation/run.py --version v4 \
  --model "outputs/length64/$LENGTH64_RUN/actor/global_step_100" \
  --server-python "$DARA_SERVER_PYTHON" --port 8100 \
  --temperature 0.6 --top-p 1 --length-min-words 64 \
  --output "outputs/eval-length64/t0p6-p1/$LENGTH64_RUN"

CUDA_VISIBLE_DEVICES=0 python evaluation/run.py --version v4 \
  --model "outputs/length64/$LENGTH64_RUN/actor/global_step_100" \
  --server-python "$DARA_SERVER_PYTHON" --port 8100 \
  --temperature 1 --top-p 1 --length-min-words 64 \
  --output "outputs/eval-length64/t1p0-p1/$LENGTH64_RUN"
```

以上为同一GPU顺序运行示例；并行执行时每份使用独立GPU、端口和上述独立输出目录。inference.json记录temperature、top_p与length_min_words，adapter将指定解码参数写入实际请求。中断后重复同一命令补齐缺失case。推理完成自动评分；score.py从inference.json读取长度阈值，CPU重评命令为 `python evaluation/score.py --version v4 --output outputs/eval-length64/t1p0-p1/<run-id>`。

Accuracy由BFCL官方checker计算。Format按既定RLLA标签结构计算。Length Reward与Length≥64对每次assistant输出计算0/1，Multi-Turn先在每题内平均，再按类别和大组聚合。Non-Live使用Simple语言宏平均后与Multiple/Parallel/ParallelMultiple等权平均；Live按1351题样本数加权；Multi-Turn对四个类别等权平均；Average是三大组等权平均。Length Reward保留0–1尺度，Length≥64以百分比展示。额外记录平均think词数。

## 汇总与交付

每份评测的summary.json中，`groups.accuracy`、`groups.format`、`groups.length`、`groups.length_ge_64`、`groups.think_words`均包含live、non_live、multi_turn、average。执行：

```bash
python evaluation/length64_results.py \
  --manifest configs/length_ge64_binary.json \
  --evaluation-root outputs/eval-length64 --output outputs/length64-report
```

`per_model.csv`包含16份逐seed、逐温度结果；`method_results.csv`包含8行方法×温度结果，报告两个seed的mean±sample SD。Accuracy/Format/Length≥64为百分比，Length Reward为0–1，think_words为词数。`three_reward_overall`为Average Accuracy/100 + Average Format/100 + Average Length。汇总按温度分别计算，并保留实际seed列表。

训练metrics.jsonl覆盖step1–100，验证覆盖step0/10/…/100。交付三路reward、总reward、response length、π、权重与活跃group数。日志中GRPO使用grpo/前缀、GDPO使用gdpo/、DARA两变体都使用dara/；依赖run ID和method区分双侧/单侧。包括 `critic/length_score/mean`、`<method>/pi_length`、`<method>/w_length`、`<method>/active_groups_length`，其他通道同理。validation字段为 `val/test_correctness/rlla`、`val/test_format/rlla`、`val/test_length/rlla`。

在训练环境导出全部曲线：

```bash
python - <<'PY'
import json
from pathlib import Path
import pandas as pd
manifest = json.loads(Path('configs/length_ge64_binary.json').read_text())
frames = []
for run in manifest['runs']:
    directory = Path('outputs/length64') / run['id']
    frame = pd.read_json(directory / 'metrics.jsonl', lines=True).sort_values('step')
    frame.to_csv(directory / 'training_dynamics.csv', index=False)
    frame['run_id'], frame['method'], frame['seed'] = run['id'], run['method'], run['seed']
    frames.append(frame)
Path('outputs/length64-report').mkdir(parents=True, exist_ok=True)
pd.concat(frames, ignore_index=True).to_csv('outputs/length64-report/training_dynamics.csv', index=False)
PY
```

交回8个run的launch.json、command.json、config.json、metrics.jsonl、training_dynamics.csv、exit.code、step100模型位置与验证输出；16份评测的inference.json、raw JSONL、官方score、diagnostics_per_case.csv、summary.json；以及汇总CSV和实际代码提交号。run目录区分`three-len64`，两档评测目录分别使用`t0p6-p1`与`t1p0-p1`。

## 给执行agent的任务文本

本分支已在现有Haotian训练环境运行`scripts/test.sh`，44项测试通过，覆盖词数阈值、缺失think标记、训练/评测长度分数一致性、完整reward通道、8-run launcher和分温度汇总。已在BFCL V4环境调用实际adapter并捕获请求，确认两档分别传入temperature=0.6/1.0、top_p=1.0、max_tokens=8192。

> 在DARA的experiment/length-ge64-binary分支按本文件和configs/length_ge64_binary.json执行。共8次1.5B训练：GRPO、GDPO、DARA双侧、DARA单侧各seeds1/2，每run四张物理GPU、G4、100steps、三奖励，length为think词数≥64得1。每个step100模型分别以T=0.6/top_p=1和T=1/top_p=1做BFCL V4评测，共16份。记录reward、π、权重、活跃group的训练动态，并交回逐seed/逐温度及方法均值和样本标准差的Accuracy、Format、Length Reward、Length≥64、think words。tmux维持任务，持续最大化使用已租GPU，下一项GPU工作先启动，再做CPU导出和提交。完成产物按本文件“汇总与交付”提交。
