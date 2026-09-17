# 8×A100 80GB SXM4：执行 agent 交接

本次机器为单节点8×NVIDIA A100 80GB SXM4，租期20小时。任务是完成 [15次1.5B训练清单](RENTAL_1P5B_20H.md)，保存全部过程模型，并完成150份checkpoint的BFCL V4评测及15份step100最终结果。唯一训练清单为 [rental_1p5b_20h.json](../configs/rental_1p5b_20h.json)。先与原集群登记run归属，再领取对应任务。

**第一调度规则：任何时候都要最大化利用所有已租GPU。有可运行的训练、checkpoint评测或必要的GPU验证任务时，可用资源立即接任务。单个任务完成后立即补位，按各路实际完成时间派发。** 所有长任务由tmux内的持续控制器维护；每分钟检查任务退出、GPU上的任务及显存占用、剩余租期和待评测模型，每30分钟记录进展与ETA。以每小时完成的有效训练和评测量衡量利用效率。

## 每个实验四卡，四个实验并行

每个实验始终使用四张物理GPU、四个FSDP rank和四路TP1 rollout。GPU0/1/2/3同时承载实验A和B，GPU4/5/6/7同时承载实验C和D；每张80GB卡承载两个独立实验的各一个训练rank。四个实验合计16个训练rank，分布在8张物理卡上，各实验的模型、梯度、optimizer和通信组相互独立。

首次派发时，A为DVAO seed4，B为GD²PO-Hard seed4，C为DVAO seed5，D为GD²PO-Hard seed5。随后按清单順序动态补位。15个run在四条实验通道上大致分配4/4/4/3次。每个run的GPU列表包含四个不同物理设备；A/B的设备列表相同，C/D的设备列表相同。

本次训练沿用现有四卡实现。512个prompt/step、G4、2048条回答/step、每次optimizer更新512条回答、每step四次更新、学习率1e-6、100steps、每10steps保存和验证。数据、reward、方法公式、动态microbatch、loss累积、KL、entropy、token whitening和rollout分配均沿用 [TRAINING.md](TRAINING.md)。

当前代码具备每个实验四rank的训练入口。本机两个四卡训练组共享设备的显存分配、进程隔离和GPU并发验证，由执行agent在新机器上落实。manifest中的target_execution记录四个共享GPU实验槽位及验证状态，正式执行时记录实际设备映射。

## 共享GPU的配置与验证

机器到手用 `nvidia-smi` 和 `nvidia-smi topo -m` 记录GPU型号、显存、编号和互联。每个实验建立独立Ray实例、临时目录、端口、训练日志及输出目录。为A/B分别启动仅可见GPU0–3的Ray实例，为C/D分别启动仅可见GPU4–7的实例；每个实例声明四张GPU，每个实验创建自己的四rank通信组。现有 `training/launch.py --ray-address`可连接独立Ray实例，执行agent在manifest启动入口接入各实例地址，并记录实际配置。核对每个实例中rank0–3映射到四张不同物理卡。

按每实验每卡约40GB的总显存预算安排两个实验共享，实测须给CUDA context、通信和训练峰值保留余量。预算同时覆盖actor、ref、optimizer、activations与rollout KV cache。当前vendored vLLM0.6.3在 `vendor/verl/verl/third_party/vllm/vllm_v_0_6_3/worker.py::determine_num_available_blocks` 中使用 `free_gpu_memory * gpu_memory_utilization` 决定KV block数量；默认0.6作用于profile时的空闲显存，两个进程的启动顺序会影响可用缓存。执行agent须按原40GB训练的KV block预算或经测量的固定缓存上限配置两个实验，记录各自KV blocks及整个训练/rollout周期的显存峰值，并确认训练batch和原rollout配置一致。

在目标机器先用一组四卡同时运行两个实验的资源验证，另一组四卡继续执行已就绪任务。验证覆盖两个独立通信组的并发collective、rollout、actor更新、validation以及step10模型导出与重新加载；启动和运行期间均监测OOM、各rank是否持续推进和整机吞吐。PyTorch对同一GPU上的多个NCCL进程提示存在死锁或invalid usage可能，见 [distributed说明](https://docs.pytorch.org/docs/stable/distributed)。因此共享显存与通信必须在实际Torch/NCCL/驱动组合下验证。若采用CUDA MPS帮助进程共享GPU，按 [NVIDIA MPS说明](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html) 配置并纳入同样的并发验证，记录实际结果。

完成一组四卡双实验的显存与通信验证后，在另一组采用相同配置，启用四个实验并行。若某项验证失败，修正该共享资源问题；其余可执行训练和BFCL任务持续运行。已开始的完整训练保留当前run直至完成。算法和训练接线发生修改时运行 `bash scripts/test.sh`，资源配置和GPU并发的实测结果随交接保存。

## 环境、启动与输出

按 [HANDOFF.md](HANDOFF.md) 安装 `rdgdpo`、`dara-inference`、`dara-bfcl-v4` 三个环境，获取固定数据和Qwen2.5-1.5B-Instruct。训练使用清单中的15个run与原seed。每个run独立保存命令、展开配置、日志、exit.code和模型，记录world_size=4、实际GPU列表、同设备上的另一实验、Ray地址和显存配置。

下面是一个现有四卡训练的启动命令。执行agent完成共享资源接线后，把四个实验各自的Ray地址和缓存预算接入同一启动流程，并用dry-run确认每个实验均为四rank及原训练参数。

```bash
conda activate rdgdpo
export DARA_MODELS=/path/to/dara-models
mkdir -p outputs/rental-1p5b/logs
CUDA_VISIBLE_DEVICES=0,1,2,3 python training/run_manifest.py \
  --manifest configs/rental_1p5b_20h.json \
  --id haotian_1p5b_dvao_s4_save10 \
  --model-root "$DARA_MODELS" --output-root outputs/rental-1p5b \
  > outputs/rental-1p5b/logs/haotian_1p5b_dvao_s4_save10.log 2>&1
```

检查CPU/RAM、共享内存和磁盘IO是否足够支撑16个训练rank以及模型导出，按实际进程需要分配。每次100step训练有10份HF模型，15次约996GiB，准备至少2TB持久空间并额外计入BFCL原始输出。任务失败时保存该attempt记录，清理该任务占用后立即补入可执行工作；训练重试使用新attempt目录。

## 训练、评测与租期衔接

调度器同时跟踪物理GPU和共享实验槽位。一个四卡实验结束后，即使同组另一个实验仍在训练，也应立即补入下一项能完整跑完的训练。剩余租期不足时，可在该组剩余显存和通信条件经过验证的情况下，给空出的份额分配checkpoint评测，并明确限制推理缓存。完全空出的GPU按每卡一份checkpoint评测派发；全部训练完成后，八卡并行评测八份checkpoint。

每个完整run优先评测step100，再补齐step10至90。BFCL vLLM服务的memory0.85默认适用于独占GPU；同卡还有训练时，需要按实际剩余资源设置推理缓存预算，并保持既定prompt、解析、解码参数和评分方式。官方评分与CSV汇总使用CPU，推理结束后及时回收GPU资源并补位。

两个实验共享GPU的算力和带宽。记录共享时每个完整run耗时T_shared与独占四卡耗时T_solo，四并行时15次训练约需4×T_shared，两并行约需8×T_solo。T_shared小于2×T_solo时，四并行能提高整体训练吞吐。ETA用实际共享负载下的step、validation和保存耗时更新，环境准备、资源验证、BFCL评测与数据回传时间分别计入租期预算。

持续控制器自动补位，每次决定是否再开训练时，使用剩余租期、最近同类完整run耗时及导出耗时。租期较短时继续补齐可增量保存的评测。交回已完成及未完成的模型清单、原始输出及路径，评测后续从相同输出目录补齐。

## BFCL V4协议与交付

执行 [EVALUATION.md](EVALUATION.md) 的完整V4协议：14类3301cases，相同ToolRL prompt与解析器，temperature0.6、top_p0.95、推理seed0、最大生成8192tokens。每个checkpoint报告Non-Live AST、Live AST、Multi-Turn、Average的Accuracy与Format，按方法、step汇总三个training seeds的均值与样本标准差。15个run全部评测steps10/20/…/100，共150份结果。**第15项DVAO seed3明确包含step100最终评测，和其他14个run一起交付最终表格。** step100评分同时作为过程曲线终点。

每张评测GPU使用独立端口和输出目录；以下为第15项示例，其余run替换DARA_RUN_ID：

```bash
conda activate dara-bfcl-v4
export DARA_SERVER_PYTHON=/path/to/envs/dara-inference/bin/python
export DARA_RUN_ID=haotian_1p5b_dvao_s3_save10
for step in 100 10 20 30 40 50 60 70 80 90; do
  CUDA_VISIBLE_DEVICES=0 python evaluation/run.py --version v4 \
    --model "outputs/rental-1p5b/$DARA_RUN_ID/actor/global_step_$step" \
    --server-python "$DARA_SERVER_PYTHON" --port 8000 \
    --output "outputs/eval-v4-checkpoints/$DARA_RUN_ID/step_$step" || break
done
python evaluation/checkpoint_results.py \
  --manifest configs/rental_1p5b_20h.json \
  --evaluation-root outputs/eval-v4-checkpoints \
  --output outputs/eval-v4-checkpoints/report
```

交回150份checkpoint的原始推理、官方评分、逐题诊断和summary.json，以及 `checkpoint_per_model.csv`、`checkpoint_method_results.csv`、`process_final_per_model.csv`、`process_final_method_results.csv`。交回15个run的训练配置、完整metrics.jsonl、training_dynamics.csv、HF模型路径、硬件布局与共享GPU验证记录。GRPO、GDPO、DARA的曲线包含correctness/format reward、各通道π、权重和活跃group数量。最终记录明确训练完成数、已评分checkpoint数和剩余任务。
