# 合作者使用说明

拿到仓库访问权限后，从下面的安装流程开始。训练设置见 [TRAINING.md](TRAINING.md)，评测设置见 [EVALUATION.md](EVALUATION.md)，已有结果见 [STATUS.md](STATUS.md)。

本分支的 Length≥64 二值三奖励实验按 [HANDOFF_LENGTH64.md](HANDOFF_LENGTH64.md) 执行：1.5B、四方法各 seeds1/2，共8次训练，step100分别以T=0.6和T=1.0评测，top_p固定1，共16次BFCL V4评测。

## 获取仓库、环境和模型

在 Linux x86_64、Conda 和 NVIDIA GPU 环境中执行：

```bash
git clone git@github.com:zhaihaotian/DARA.git
cd DARA
bash environments/install_training.sh rdgdpo
conda activate rdgdpo
python scripts/fetch_data.py
export DARA_MODELS="$PWD/../dara-models"
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir "$DARA_MODELS/Qwen2.5-1.5B-Instruct"
hf download Qwen/Qwen2.5-3B-Instruct --local-dir "$DARA_MODELS/Qwen2.5-3B-Instruct"
```

安装脚本会执行依赖检查和训练单元测试。已有本地模型时，将 `DARA_MODELS` 设为包含这两个模型目录的路径。

当前最终比较补齐与1.5B过程checkpoint实验见 [CHECKPOINT_STUDY.md](CHECKPOINT_STUDY.md)，使用独立manifest及每10steps保存设置。8×A100 40GB租机的独立agent交接见 [RENTAL_A100_40G_AGENT.md](RENTAL_A100_40G_AGENT.md)，15次训练清单见 [RENTAL_1P5B_20H.md](RENTAL_1P5B_20H.md)。该清单要求每个run完成steps10/20/…/100的BFCL V4评测，并交付过程数据和step100最终结果表；第15项DVAO seed3同样包含最终评测。

## 领取实验

| 实验 | 模型 | 方法 | G | Seeds | 工作量 |
|---|---|---|---|---|---|
| Group size | 1.5B | GRPO、GDPO、DARA | 4/8/16/32 | 0/1/2/4/5 | G4复用15个完整run，G8/16/32新增45个 |
| 三奖励 | 1.5B、3B | GRPO、GDPO、DARA | 4 | 0/1/2/4/5 | 新增30个run |

Group实验清单位于 `configs/group_ablation.json`，三奖励位于 `configs/three_rewards.json`。从 `status=planned` 的条目选择run ID，并在协作记录里登记负责人。每run固定4×A10040GB、100steps；优先覆盖三个方法的seeds0/1/2，之后做4/5。按四卡一组并行使用已获配GPU。

Group实验固定2048回答/step和每step四次optimizer更新，脚本根据G同步设置prompt batch、PPO minibatch与microbatch。三奖励在原correctness、format基础上加入length，固定G4；其余参数沿用相同配置。

## 启动训练

进入分配到四张GPU的计算节点，打开持久终端：

```bash
tmux new -s dara-training
conda activate rdgdpo
cd /path/to/DARA
export DARA_MODELS=/path/to/dara-models
mkdir -p outputs/logs
python training/run_manifest.py --manifest configs/group_ablation.json \
  --id 1p5b-dara-g8-s0-two --model-root "$DARA_MODELS" --output-root outputs \
  > outputs/logs/1p5b-dara-g8-s0-two.log 2>&1
```

在tmux中按 `Ctrl-b d` 脱离，之后用 `tmux attach -t dara-training` 返回。脚本继承调度器提供的GPU可见范围。

运行三奖励时选择对应manifest和run ID：

```bash
python training/run_manifest.py --manifest configs/three_rewards.json \
  --id 3b-dara-g4-s0-three --model-root "$DARA_MODELS" --output-root outputs \
  > outputs/logs/3b-dara-g4-s0-three.log 2>&1
```

命令加 `--dry-run` 可查看完整配置。每run使用独立输出目录，记录 `launch.json`、`command.json`、`config.json`、`metrics.jsonl` 和 `exit.code`；最终模型在 `outputs/<run-id>/actor/global_step_100`。

三奖励的长度分数为think区域词数/512、round2、上限1，训练日志记录 `critic/length_score/mean`；DARA还记录 `dara/pi_length` 与 `dara/w_length`。Group实验和三奖励的全部固定参数见 [TRAINING.md](TRAINING.md)。

## 训练 dynamics

每个run的 `outputs/<run-id>/metrics.jsonl` 按step保存训练指标。所有方法记录correctness、format、length的reward统计、总reward和回答token数；GRPO、GDPO、DARA同时记录各奖励通道的π、权重和活跃group数量。

| 曲线 | 日志字段 |
|---|---|
| Correctness reward | `critic/correctness_score/mean` |
| Format reward | `critic/format_score/mean` |
| Length reward | `critic/length_score/mean` |
| 总reward | `critic/score/mean` |
| 平均回答token数 | `response_length/mean` |
| 通道密度π | `<method>/pi_correctness`、`<method>/pi_format`；三奖励另有 `<method>/pi_length` |
| 通道权重 | `<method>/w_correctness`、`<method>/w_format`；三奖励另有 `<method>/w_length` |
| 活跃group数 | `<method>/active_groups_correctness`、`<method>/active_groups_format`；三奖励另有 `<method>/active_groups_length` |

`<method>` 对应 `grpo`、`gdpo` 或 `dara`，例如 `gdpo/pi_format`。三个方法的π都按原始通道reward分别计算组内z-score、广播到有效response tokens后，统计优势绝对值和大于1e-8的prompt group比例。GRPO的日志权重是原始reward求和时各通道的系数1；GDPO是各通道标准化优势相加时的系数1；DARA记录实际使用的密度校准权重。

训练指标覆盖steps1–100；验证指标在steps0/10/…/100记录为 `val/test_correctness/rlla`、`val/test_format/rlla` 和 `val/test_length/rlla`。对应的逐条验证回答与reward保存在 `outputs/<run-id>/eval/step_000.jsonl`、`step_010.jsonl` 等文件中。

在训练环境中将单个run的完整曲线导出为CSV，供画reward、π和权重随step变化的图：

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

run = Path('outputs/3b-dara-g4-s0-three')
metrics = pd.read_json(run / 'metrics.jsonl', lines=True)
metrics.sort_values('step').to_csv(run / 'training_dynamics.csv', index=False)
PY
```

## 评测最终模型

创建推理服务和BFCL环境：

```bash
bash environments/install_inference.sh dara-inference
bash environments/install_evaluation.sh v3 dara-bfcl-v3
bash environments/install_evaluation.sh v4 dara-bfcl-v4
```

记录推理环境Python路径后，运行V3与V4：

```bash
conda activate dara-inference
export DARA_SERVER_PYTHON="$(command -v python)"
conda activate dara-bfcl-v3
CUDA_VISIBLE_DEVICES=0 python evaluation/run.py --version v3 \
  --model outputs/1p5b-dara-g8-s0-two/actor/global_step_100 \
  --server-python "$DARA_SERVER_PYTHON" --port 8000 \
  --output outputs/eval-v3/1p5b-dara-g8-s0-two
conda activate dara-bfcl-v4
CUDA_VISIBLE_DEVICES=0 python evaluation/run.py --version v4 \
  --model outputs/1p5b-dara-g8-s0-two/actor/global_step_100 \
  --server-python "$DARA_SERVER_PYTHON" --port 8000 \
  --output outputs/eval-v4/1p5b-dara-g8-s0-two
```

每个checkpoint评测使用一张GPU。并行任务分别设置GPU、端口和输出目录。推理结果逐case保存，重复命令可补齐缺失case；评分完成后查看 `summary.json`。具体子集、Average和Average Format的定义见 [EVALUATION.md](EVALUATION.md)。

三奖励实验也使用上面的评测命令。评分会同时计算Accuracy、Format、Length Reward、Length≥512和平均think词数。Length Reward沿用训练中的长度公式；Length≥512统计think文本达到512个空白分隔词的比例。这些逐case指标写入 `diagnostics_per_case.csv`，分组结果写入 `summary.json`。

从 `summary.json` 的 `groups` 读取表格数据：`accuracy` 和 `format` 下分别是各组准确率和格式分数，`length` 是Length Reward，`length_ge_512` 是长度达标比例，`think_words` 是平均think词数。V3的组为 `non_live`、`live`；V4再包含 `multi_turn`、`average`。例如V4 Average、Average Format和Average Length分别读取 `groups.accuracy.average`、`groups.format.average` 和 `groups.length.average`。Accuracy、Format与Length≥512乘100后以百分数展示，Length Reward保留0–1尺度。按相同模型规模、方法、G和奖励设置汇总各seed，得到表格里的mean±sample SD。

## 交回结果

交回训练配置、命令、完整metrics.jsonl、training_dynamics.csv、最终模型位置，以及评测的raw JSONL、官方score、逐case指标和summary.json。三奖励的结果表包含Length Reward、Length≥512和平均think词数。完成的训练记录包含steps1–100、validation steps0/10/…/100和可加载的最终模型。汇总按checkpoint分别计算指标，再报告跨seed的mean±sample SD；筛选分析附上使用的seed列表与n。

已有42个训练run、44个模型的评测结果及训练曲线数据在 `results/reference/`。G4对照直接复用其中对应结果。DARA历史日志包含π；GRPO/GDPO历史密度字段留空。各方法已有结果和待执行任务见 [STATUS.md](STATUS.md)。

## 发给执行 agent 的任务文本

> 请按照本仓库 docs/HANDOFF.md 安装环境、下载固定数据和模型，并运行分配给你的manifest条目。每run使用4张A10040GB、100steps和manifest规定的参数；Group实验固定2048回答/step，三奖励实验使用correctness+format+length。用tmux运行，每run单独保存日志和输出。完成训练后对step100模型执行本仓库的BFCL V3与V4评测，按相同设置汇总各seed的mean±sample SD，三奖励结果包含Length Reward、Length≥512和平均think词数。交回完整训练日志、training_dynamics.csv、模型路径、逐case输出及表格数据；曲线数据包含各路reward，以及GRPO、GDPO、DARA各通道的π、权重和活跃group数。优先覆盖各方法的seeds0/1/2，再完成4/5。
