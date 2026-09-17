# DARA

独立整理的 Haotian Tool Calling 训练与评测仓库。DARA 对应历史实验中的 **RD-GDPO symmetric**，使用原 Haotian / NVIDIA GDPO 风格的训练 infrastructure。实现包含 GRPO、GDPO、DARA、RVPO、DVAO、GDPO-SAW、GD²PO-Hard。

本仓库提供固定参数训练入口、实际运行环境的依赖锁定、原始训练数据的固定版本下载入口、BFCL V3/V4 推理与评分脚本，以及已完成实验的逐 seed 结果和逐步训练记录。模型权重和大型 BFCL rollout 文件保留在原实验存储；定位信息见 [数据与工件](docs/ARTIFACTS.md)。

## 从哪里开始

| 需求 | 文档或入口 |
|---|---|
| 合作者从 clone 到训练、评测的完整流程 | [HANDOFF.md](docs/HANDOFF.md)、[AGENTS.md](AGENTS.md) |
| 8×A100 40GB租机执行agent交接 | [RENTAL_A100_40G_AGENT.md](docs/RENTAL_A100_40G_AGENT.md) |
| 可直接用于论文的实验设置与评测协议 | [PAPER_EXPERIMENTS.md](docs/PAPER_EXPERIMENTS.md) |
| 创建环境 | [ENVIRONMENT.md](docs/ENVIRONMENT.md) |
| 固定训练设置与方法定义 | [TRAINING.md](docs/TRAINING.md)、[ALGORITHMS.md](docs/ALGORITHMS.md) |
| BFCL V3/V4、Avg、Format、Length 的完整定义 | [EVALUATION.md](docs/EVALUATION.md) |
| 已完成实验和后续任务 | [STATUS.md](docs/STATUS.md) |
| 所有 seed 的最终评测表 | [ALL_SEEDS.md](results/reference/ALL_SEEDS.md) |
| 事后筛选的诊断表 | [SELECTED_SEEDS.md](results/reference/SELECTED_SEEDS.md) |
| 自己画训练曲线 | [training_dynamics.csv](results/reference/training_dynamics.csv) |
| Group 消融的可执行矩阵 | [group_ablation.json](configs/group_ablation.json) |
| 三奖励的可执行矩阵 | [three_rewards.json](configs/three_rewards.json) |

## 安装与运行

在 Linux x86_64、NVIDIA GPU 和 Conda 环境中：

```bash
bash environments/install_training.sh rdgdpo
conda activate rdgdpo
python scripts/fetch_data.py
python training/launch.py --method dara --seed 0 \
  --model-size 1.5b --model /models/Qwen2.5-1.5B-Instruct \
  --output outputs/1p5b-dara-g4-s0-two --dry-run
```

实际训练在已有 **4 × A100 40GB** allocation 内执行同一命令，去掉 `--dry-run`。每个 run 100 steps，最终模型位于 `outputs/<run>/actor/global_step_100`。每个 run 使用独立输出目录。

```bash
python training/run_manifest.py --manifest configs/group_ablation.json \
  --id 1p5b-dara-g8-s0-two --model-root /models --output-root outputs
python training/run_manifest.py --manifest configs/three_rewards.json \
  --id 3b-dara-g4-s0-three --model-root /models --output-root outputs
```

评测必须使用单独的 BFCL 环境和推理环境，完整命令见 [EVALUATION.md](docs/EVALUATION.md)。训练测试用 `bash scripts/test.sh`。当前交接的数值验证及其范围见 [VALIDATION.md](docs/VALIDATION.md)。

## 来源

训练源码 vendored 于 `vendor/verl`，源自 Haotian 的 RD-GDPO 工作目录，沿用该训练环境和实现。附加算法继承已有的 Jay→Haotian 移植。版本与差异见 [ALGORITHMS.md](docs/ALGORITHMS.md)。许可证和第三方来源见 [LICENSE](LICENSE)、[THIRD_PARTY.md](THIRD_PARTY.md)。
