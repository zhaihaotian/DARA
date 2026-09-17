# 环境重建

## 实际训练环境

2026-09-17 核对服务器 Conda 列表、3B 的实际启动脚本及已完成 run 的 environment.txt 后确认：Haotian 训练使用 `/users/2/lian0190/miniconda3/envs/gdpo-paper/bin/python`。新安装脚本默认把同一套依赖创建为 **rdgdpo**；名称不改变计算行为。不要把 Jay 训练环境用于本仓库训练。

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

完整已安装分发版本见 `environments/training.json`，可安装锁定见 `environments/training.lock.txt`。锁定文件不含服务器本地 wheel 路径；FlashAttention 使用与现有环境相同的公开预编译 wheel（cu123、torch2.4、CXX11 ABI false、CPython3.10）。PyTorch 本身仍是 cu121。安装脚本指定全部依赖的确切版本，使用 `--no-deps` 避免 resolver 改动旧训练栈；现有训练环境 `pip check` 已通过，安装脚本也会执行该检查。

```bash
bash environments/install_training.sh rdgdpo
conda activate rdgdpo
bash scripts/test.sh
```

目标是 Linux x86_64 + NVIDIA A100 40GB。GPU 驱动需支持所装 CUDA runtime；不要在 Mac 或无 CUDA 平台期待同一个 wheel 工作。服务器原有环境保留，新环境单独创建。当前交接验证使用现有环境；未声称已经在另一台全新机器完整安装成功。

## BFCL 推理与评分环境

历史 BFCL vLLM 服务使用服务器上名为 `gdpo-jay` 的 Python 环境，但被加载的 checkpoint 是 Haotian 训练的模型。这个环境名并不表示评测改用了 Jay 训练 infra。实际推理栈为 Python3.12.14、Torch2.8.0、vLLM0.11.0、Transformers4.57.3。BFCL 不适合直接装进旧训练环境。

```bash
bash environments/install_inference.sh dara-inference
bash environments/install_evaluation.sh v3 dara-bfcl-v3
bash environments/install_evaluation.sh v4 dara-bfcl-v4
```

V3 固定 `bfcl-eval==2025.8.6.2`。V4 固定 gorilla 源码提交 `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`，通过环境的 `.pth` 指向该源码，这是历史评测采用的加载方式。V4 的来源应看源码路径与提交，不能根据继承依赖里的旧 `bfcl_eval` distribution metadata 判断。其 upstream pyproject 的可选远程 provider 依赖与实际本地评测环境并非逐项一致，因此按实际依赖快照安装，不自动升级整套 BFCL 依赖。

`inference.json` / `bfcl-v3.json` / `bfcl-v4.json` 保存完整历史分发清单；对应 lock 文件固定执行依赖。独立 `flash_attn` wheel 未用于该 vLLM API 服务，推理安装依赖 vLLM 自带的 attention backend。训练环境则必须安装上面指定的 FlashAttention wheel。

模型下载后固定为本地目录；所有算法使用同一份 Qwen2.5-1.5B-Instruct 或 Qwen2.5-3B-Instruct 权重和 tokenizer。脚本继承调度器的 `CUDA_VISIBLE_DEVICES`，不硬编码服务器 GPU 编号。
