# 环境重建

## 实际训练环境

Haotian 已完成实验使用 `/users/2/lian0190/miniconda3/envs/gdpo-paper/bin/python`。新安装脚本默认把同一套依赖创建为 **rdgdpo**。

| 组件 | 实际版本 |
|---|---|
| Python | 3.10.21 |
| PyTorch | 2.4.0+cu121 |
| vLLM | 0.6.3 |
| Ray | 2.10.0 |
| Transformers | 4.47.1 |
| FlashAttention | 2.6.3 |
| xFormers | 0.0.27.post2 |
| TensorDict | 0.5.0 |
| NumPy | 1.26.4 |
| Hydra / OmegaConf | 1.3.6 / 2.3.1 |

完整已安装分发版本见 `environments/training.json`，可安装锁定见 `environments/training.lock.txt`。FlashAttention 使用与现有环境相同的公开预编译 wheel（cu123、torch2.4、CXX11 ABI false、CPython3.10）。PyTorch 本身仍是 cu121。安装脚本通过完整依赖锁定和 `--no-deps` 安装指定版本；现有训练环境 `pip check` 已通过，安装脚本也会执行该检查。

```bash
bash environments/install_training.sh rdgdpo
conda activate rdgdpo
bash scripts/test.sh
```

运行平台为 Linux x86_64 + NVIDIA A100 40GB，GPU 驱动支持对应 CUDA runtime。新环境独立创建，验证记录见 [VALIDATION.md](VALIDATION.md)。

## BFCL 推理与评分环境

BFCL vLLM 服务的历史环境名为 `gdpo-jay`，加载 Haotian 训练的 checkpoint。推理栈为 Python3.12.14、Torch2.8.0、vLLM0.11.0、Transformers4.57.3。评测环境单独创建。

```bash
bash environments/install_inference.sh dara-inference
bash environments/install_evaluation.sh v3 dara-bfcl-v3
bash environments/install_evaluation.sh v4 dara-bfcl-v4
```

V3 固定 `bfcl-eval==2025.8.6.2`。V4 固定 gorilla 源码提交 `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`，通过环境的 `.pth` 指向该源码，这是历史评测采用的加载方式。V4 的 evaluator 版本由这个提交和源码路径确定，安装脚本使用实际运行环境的依赖快照。

`inference.json` / `bfcl-v3.json` / `bfcl-v4.json` 保存完整历史分发清单；对应 lock 文件固定执行依赖。推理服务使用 vLLM 自带的 attention backend，训练使用上面指定的 FlashAttention wheel。

模型下载后固定为本地目录；所有算法使用同一份 Qwen2.5-1.5B-Instruct 或 Qwen2.5-3B-Instruct 权重和 tokenizer。脚本继承调度器的 `CUDA_VISIBLE_DEVICES`。
