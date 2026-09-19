# 3B DVAO 与 GD²PO-Hard 补种子交接

## 任务

使用 Haotian training infrastructure 和 Qwen2.5-3B-Instruct 完成下面四个 run：

| Run ID | 方法 | Seed |
|---|---|---:|
| `haotian_3b_dvao_s1_save10` | DVAO | 1 |
| `haotian_3b_dvao_s2_save10` | DVAO | 2 |
| `haotian_3b_gd2po_hard_s1_save10` | GD²PO-Hard | 1 |
| `haotian_3b_gd2po_hard_s2_save10` | GD²PO-Hard | 2 |

固定设置为 G=4、correctness+format 两奖励、100 training steps、每10 steps验证并保存一次 Hugging Face actor。数据、学习率、batch、rollout、KL、entropy、loss累积和随机种子口径全部使用仓库现有配置。任务定义已经写入 [`configs/checkpoint_study.json`](../configs/checkpoint_study.json)。

已经完成的 DVAO seed0 与 GD²PO-Hard seed0 均保存了 `global_step_10/20/.../100` 的完整 actor。它们的过程 BFCL V4 评测由现有本地队列执行。本次四个新增 run 保存相同的过程模型，最终比较只需评测各自的 step100。

## 环境

```bash
git clone git@github.com:zhaihaotian/DARA.git
cd DARA
bash environments/install_training.sh rdgdpo
conda activate rdgdpo
python scripts/fetch_data.py
export DARA_MODELS=/path/to/dara-models
hf download Qwen/Qwen2.5-3B-Instruct \
  --local-dir "$DARA_MODELS/Qwen2.5-3B-Instruct"
export DARA_OUTPUT=/path/to/dara-3b-runs
mkdir -p "$DARA_OUTPUT/logs"
```

服务器已有相同版本环境时可以直接使用；实际训练依赖版本见 [`ENVIRONMENT.md`](ENVIRONMENT.md)。开始前执行仓库测试：

```bash
bash scripts/test.sh
```

## 四张 A100 40GB 启动

每个 run 独占四张 A100 40GB。八张卡分成 `0,1,2,3` 与 `4,5,6,7` 两组，同时运行两个任务；其中一个结束后立即补入下一个任务。

```bash
tmux new-session -d -s dvao-s1 \
  "cd '$PWD' && conda run --no-capture-output -n rdgdpo env CUDA_VISIBLE_DEVICES=0,1,2,3 \
   python training/run_manifest.py --manifest configs/checkpoint_study.json \
   --id haotian_3b_dvao_s1_save10 --model-root '$DARA_MODELS' \
   --output-root '$DARA_OUTPUT' > '$DARA_OUTPUT/logs/haotian_3b_dvao_s1_save10.log' 2>&1"

tmux new-session -d -s hard-s1 \
  "cd '$PWD' && conda run --no-capture-output -n rdgdpo env CUDA_VISIBLE_DEVICES=4,5,6,7 \
   python training/run_manifest.py --manifest configs/checkpoint_study.json \
   --id haotian_3b_gd2po_hard_s1_save10 --model-root '$DARA_MODELS' \
   --output-root '$DARA_OUTPUT' > '$DARA_OUTPUT/logs/haotian_3b_gd2po_hard_s1_save10.log' 2>&1"
```

第一轮完成后，以相同方式把 run ID 换成 `haotian_3b_dvao_s2_save10` 和 `haotian_3b_gd2po_hard_s2_save10`。四卡 A100 的完整 run 申请至少6小时，并为模型导出预留时间。

## 两张 H100 启动

每个 run 使用两张 H100，各物理 rank 顺序执行两路逻辑分组，训练计算口径与原四卡 A100 设置一致。每对 H100 先运行 GPU 对照：

```bash
CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n rdgdpo \
  python training/check_logical_dp.py --world-size 2 --device cuda \
  --reference training/reference/four_rank_ppo.json \
  --output "$DARA_OUTPUT/two_gpu_validation_01.json"
```

对照通过后启动训练：

```bash
CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n rdgdpo \
  python training/launch.py --method dvao --seed 1 --model-size 3b \
  --model "$DARA_MODELS/Qwen2.5-3B-Instruct" --group-size 4 --rewards two \
  --save-freq 10 --gpus 2 \
  --output "$DARA_OUTPUT/haotian_3b_dvao_s1_save10"
```

其他三个 run 分别替换 `--method`、`--seed` 和输出目录。DVAO 使用 `--method dvao`，GD²PO-Hard 使用 `--method gd2po_hard`。八张 H100 可以分成四组，同时启动四个 run；四张 H100 同时启动两个。两卡逻辑四路的实现与数值验证见 [`TWO_GPU_LOGICAL_FOUR.md`](TWO_GPU_LOGICAL_FOUR.md)。根据现有完整运行记录，每对 H100 为单个 run 预留至少11小时。

## 完成检查与交付

每个输出目录必须包含 `launch.json`、`command.json`、`config.json`、完整的 `metrics.jsonl`、训练日志，以及下面十个可加载 actor：

```text
actor/global_step_10
actor/global_step_20
actor/global_step_30
actor/global_step_40
actor/global_step_50
actor/global_step_60
actor/global_step_70
actor/global_step_80
actor/global_step_90
actor/global_step_100
```

训练完成后为每个 run 导出 dynamics：

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path('/path/to/dara-3b-runs')
for run in (
    'haotian_3b_dvao_s1_save10',
    'haotian_3b_dvao_s2_save10',
    'haotian_3b_gd2po_hard_s1_save10',
    'haotian_3b_gd2po_hard_s2_save10',
):
    path = root / run
    frame = pd.read_json(path / 'metrics.jsonl', lines=True).sort_values('step')
    frame.to_csv(path / 'training_dynamics.csv', index=False)
PY
```

交回四个输出目录、硬件信息、每个 run 的起止时间和 `training_dynamics.csv`。训练输出同步完成后，再释放训练机器。

## BFCL V4 最终评测

每个新增 run 只评测 `actor/global_step_100`。按照 [`EVALUATION.md`](EVALUATION.md) 创建推理和 V4 环境，然后为四个模型分别运行：

```bash
conda activate dara-bfcl-v4
CUDA_VISIBLE_DEVICES=0 python evaluation/run.py --version v4 \
  --model "$DARA_OUTPUT/haotian_3b_dvao_s1_save10/actor/global_step_100" \
  --server-python /path/to/envs/dara-inference/bin/python \
  --port 8000 --output "$DARA_OUTPUT/eval-v4/haotian_3b_dvao_s1_save10"
```

每张空闲 GPU 运行一个 checkpoint，分别使用独立端口和输出目录。交回 `raw/`、官方评分目录、`diagnostics_per_case.csv` 和 `summary.json`。最终表按 DVAO seeds0/1/2 与 GD²PO-Hard seeds0/1/2 分别计算 mean±sample SD。

