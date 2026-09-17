# 四卡实验的跨节点布局

跨节点使用两台A100节点、每台两张GPU，一个实验的总GPU数保持4。Haotian的FSDP分片跨四个rank工作，vLLM保持TP=1，在四张卡上分别生成回答。G4时继续使用512 prompts、2048 responses/step、PPO minibatch128、microbatch64、学习率1e-6和100 steps；每step四次optimizer更新、名义梯度累积次数2保持相同。过程模型保存使用 `trainer.save_freq=10`，验证使用 `trainer.test_freq=10`。

2026-09-17读取的MSI实时Slurm配置允许 `a100-4` 最多4个节点；`a100-8` 和 `msigpu` 当前每job最多1个节点。两节点A100应放进同一份Slurm申请，同时获配、共同使用同一租期。下面的申请参数已通过调度器检查：

```bash
sbatch --test-only --account=myersc --partition=a100-4 \
  --nodes=2 --ntasks-per-node=1 --cpus-per-task=8 \
  --mem=96G --gres=gpu:a100:2 --tmp=100G \
  --time=06:00:00 --time-min=05:30:00 --wrap=true
```

这里CPU、内存、GPU和临时磁盘按每节点申请，合计16CPU、192GB内存、4张A100。`--test-only`检查申请参数并返回当时的调度估计；正式训练由同样资源设置的Slurm脚本启动。

获配后，两个节点读取共享文件系统中的相同代码、Conda环境、模型和数据。脚本通过 `scontrol show hostnames "$SLURM_JOB_NODELIST"` 获取节点列表，选择第一台为Ray head。在各自的Slurm step内，head执行 `ray start --head --node-ip-address=<head-IP> --port=<job-port> --num-cpus=8 --num-gpus=2 --block`，另一台执行 `ray start --address=<head-IP>:<job-port> --num-cpus=8 --num-gpus=2 --block`。启动环境在两个节点上统一传入训练seed、reward开关、PYTHONPATH和vLLM设置。Ray按Slurm提供的可见GPU运行，端口和本地临时目录按job分配。

Ray登记两台节点、总计4张GPU后，在head启动一次训练driver，并连接上述Ray地址。训练配置设为 `trainer.nnodes=2`、`trainer.n_gpus_per_node=2`。底层 [main_ppo.py](../vendor/verl/verl/trainer/main_ppo.py) 由这两个配置生成 `[2, 2]` 资源池；[RayWorkerGroup](../vendor/verl/verl/single_controller/ray/base.py) 分配全局rank0–3与本地rank0–1；[FSDP worker](../vendor/verl/verl/workers/fsdp_workers.py) 用NCCL建立四卡通信。checkpoint由现有保存逻辑写到共享输出目录。这与[verl多节点文档](https://verl.readthedocs.io/en/v0.3.x/start/multinode.html)中的Ray head/worker流程一致。

本仓库当前 [training/launch.py](../training/launch.py) 固定 `nnodes=1`、`n_gpus_per_node=4`，并清除外部 `RAY_ADDRESS`。接通跨节点入口需要让启动器接受节点布局和明确的Ray地址，同时保留单节点默认值，并加入上述Slurm/Ray启动脚本。环境中的Ray2.10支持通过 `RAY_ADDRESS` 连接现有集群，底层训练算法可沿用现有实现。

实际首跑需确认Ray识别两个节点、四个rank完成一次NCCL通信，并完成训练更新与step10模型保存。通信测试用来发现地址或节点间网络不可达的问题；训练更新和保存用于确认FSDP、vLLM以及共享输出路径接通。目前已完成代码路径核对和Slurm参数检查，双节点GPU实跑尚待执行。

这种布局的用途是利用分散在两个节点的空闲GPU。相比同节点四卡，FSDP增加跨节点参数和梯度传输；对等待时间的改善与每步训练耗时应分别记录，使用首次实跑数据决定后续排队策略。
