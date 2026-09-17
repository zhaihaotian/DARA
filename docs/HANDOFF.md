# 合作者使用说明

拿到仓库访问权限后，从下面的安装流程开始。训练设置见 [TRAINING.md](TRAINING.md)，评测设置见 [EVALUATION.md](EVALUATION.md)，已有结果见 [STATUS.md](STATUS.md)。

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

## 交回结果

交回训练配置、命令、完整metrics.jsonl、最终模型位置，以及评测的raw JSONL、官方score、逐case指标和summary.json。完成的训练记录包含steps1–100、validation steps0/10/…/100和可加载的最终模型。汇总按checkpoint分别计算指标，再报告跨seed的mean±sample SD；筛选分析附上使用的seed列表与n。

已有42个训练run、44个模型的评测结果及训练曲线数据在 `results/reference/`。G4对照直接复用其中对应结果。DARA历史日志包含π；GRPO/GDPO历史密度字段留空。各方法已有结果和待执行任务见 [STATUS.md](STATUS.md)。

## 发给执行 agent 的任务文本

> 请按照本仓库 docs/HANDOFF.md 安装环境、下载固定数据和模型，并运行分配给你的manifest条目。每run使用4张A10040GB、100steps和manifest规定的参数；Group实验固定2048回答/step，三奖励实验使用correctness+format+length。用tmux运行，每run单独保存日志和输出。完成训练后对step100模型执行本仓库的BFCL V3与V4评测，交回完整训练日志、模型路径、逐case输出及汇总结果。优先覆盖各方法的seeds0/1/2，再完成4/5。
