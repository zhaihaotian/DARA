# 两份两卡 allocation 合并训练

通过Ray把两份已获配的两卡Slurm allocation组成一个四卡实验。两份资源使用相同GPU类型，各自在自己的Slurm step内运行Ray；训练driver在第一份资源内启动一次。FSDP覆盖四个rank，vLLM保持TP=1。G4时继续使用512 prompts、2048 responses/step、PPO minibatch128、microbatch64、学习率1e-6和100 steps，每step四次optimizer更新、名义梯度累积次数2。

两份allocation的申请和到期时间独立。实验的可用时间取两个EndTime中较早的一个；启动器读取Slurm实际分配，确认两份资源都是已运行的单节点两卡、卡型相同、剩余租期满足要求。跨节点的参数和梯度通信由NCCL负责，环境、代码、数据、模型和输出目录放在两台机器可访问的共享文件系统上。

## 资源和启动

每份资源申请2张同型GPU、8CPU和96GB内存。在两个账户下可分别申请A100或H100，租期使用6、12、23小时，TimeMin为5小时30分钟。A100可选 `a100-4,a100-4-profile,a100-8,msigpu`，H100使用 `msigpu`。这些参数已通过MSI调度器检查。两份allocation可以来自不同账户。

训练环境固定Ray2.10.0与Click8.2.1，版本见 `environments/training.lock.txt`。启动器通过Ray原生集群地址连接head与worker。

在登录节点激活训练环境并打开tmux，用两份实际job ID执行通信检查：

```bash
conda activate rdgdpo
tmux new -s dara-pair-check
python training/launch_pair.py --jobs JOB_A JOB_B --output outputs/pair-check-JOB_A-JOB_B --check-only
```

检查启动两个Ray节点，各分配两张GPU。四个测试rank使用NCCL求和，输入分别为1、2、3、4；各rank应返回10。结果写入 `communication.json`，同时记录节点、GPU和设备类型。检查结束后释放本次Slurm step的占用，原allocation继续保留。

训练使用一个新的输出目录，例如：

```bash
python training/launch_pair.py --jobs JOB_A JOB_B --output outputs/1p5b-dvao-g4-s1-two-save10 --method dvao --seed 1 --model-size 1.5b --model /shared/models/Qwen2.5-1.5B-Instruct --save-freq 10
```

每份allocation默认需至少剩余5小时30分钟。确定实测耗时后，可以通过 `--minimum-seconds` 设置这次启动所需的共同剩余时间。`--save-freq 10` 在steps10/20/…/100保存HF模型权重和tokenizer，验证保持steps0/10/…/100。其他方法使用同一入口，训练参数由 [launch.py](../training/launch.py) 生成。

启动器先做四卡通信检查，通过后开始训练。`pair.json` 保存两份allocation、共同到期时间和启动环境；`head.log`、`worker.log`记录两侧日志；`metrics.jsonl`、验证回答和 `actor/global_step_<step>` 沿用现有输出格式。任一侧退出时，配对启动器结束这次运行并清理所属Ray进程，两份allocation仍由Slurm管理。

## 接线与验证

[launch_pair.py](../training/launch_pair.py) 为两份allocation分别创建 `srun --jobid=... --gres=gpu:<type>:2`，由 [ray_node.py](../training/ray_node.py) 启动Ray head与worker。两侧统一使用训练seed、reward开关、PYTHONPATH和vLLM设置。Ray端口与本地临时目录按本次启动分配；训练driver使用明确的Ray地址。

训练配置为 `trainer.nnodes=2`、`trainer.n_gpus_per_node=2`。底层 [main_ppo.py](../vendor/verl/verl/trainer/main_ppo.py) 生成 `[2, 2]` 资源池，[RayWorkerGroup](../vendor/verl/verl/single_controller/ray/base.py) 分配四个全局rank，[FSDP worker](../vendor/verl/verl/workers/fsdp_workers.py) 建立NCCL通信。独立四卡启动继续使用单节点默认设置。流程参考[verl多节点文档](https://verl.readthedocs.io/en/v0.3.x/start/multinode.html)和[Ray的Slurm说明](https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html)，命令使用当前环境的Ray2.10接口。

38项CPU测试已通过，覆盖两节点与单节点的训练参数对照、Slurm资源解析和双侧启动参数。2026-09-18在agb04与agb02各两张A100上完成四rank NCCL通信检查，各rank均返回10，启动器退出码为0。结果保存在 `/scratch.global/lian0190/DARA/audits/20260918_pair_check/communication.json`。双节点训练更新与模型导出仍待完整训练验证。首跑记录排队等待时间、每step耗时和模型保存耗时，用于后续租期安排。
