# 8×A100 80GB SXM4：执行 agent 交接

本次机器为单节点8×NVIDIA A100 80GB SXM4，租期20小时。任务是完成 [15次1.5B训练清单](RENTAL_1P5B_20H.md)，保存全部过程模型，并完成150份checkpoint的BFCL V4评测及15份step100最终结果。唯一训练清单为 [rental_1p5b_20h.json](../configs/rental_1p5b_20h.json)。先与原集群登记run归属，再领取对应任务。

**第一调度规则：任何时候都要最大化利用所有已租GPU。有可运行的训练、checkpoint评测或必要的GPU验证任务时，空闲GPU立即接任务。单个任务完成后立即补位，按各路实际完成时间派发。** 所有长任务由tmux内的持续控制器维护；每分钟检查任务退出、可用GPU、剩余租期和待评测模型，每30分钟记录进展与ETA。八卡全部有工作时，优先选择每小时完成训练及评测最多的配置。

## 资源布局与当前代码状态

目标为两张物理卡运行一个训练实验，同时保留原来的四路逻辑数据分组。GPU0/1、2/3、4/5、6/7构成四条训练通道，最多同时运行四个实验。15个run在四条通道上大致分配4/4/4/3次，完成一项立即领取下一项。先派发DVAO seed4、GD²PO-Hard seed4、DVAO seed5、GD²PO-Hard seed5，再按清单顺序派发其余run。

当前主分支的正式训练路径为四个物理rank。`training/launch.py --gpus 2`支持普通两rank训练；本次要求的四路逻辑分组接线仍需实现和GPU验证。执行agent先完成下一节的两卡实现及对照，通过后启用四路并行，并把manifest中实际使用的gpu_groups和每run的gpus同步为两卡。启动入口 `training/run_manifest.py`还需要读取并传递每run的物理GPU数和对应逻辑分组配置，实际展开配置须记录二者。

准备代码时，以GPU0–3、4–7运行两个现有四卡实验。需要GPU对照时，在一组训练自然完成后使用该组进行验证，另一组继续训练；验证结束即补入正式任务。已经开始的训练完成当前run再切换布局。若两卡版的显存、训练等效性或整机吞吐尚未达到要求，继续四卡两路运行，并把空出的卡接到BFCL评测。机器到手先用 `nvidia-smi` 和 `nvidia-smi topo -m` 记录实际GPU型号、显存、编号和互联，为并发任务划分互不重叠的GPU范围。

## 两卡保留四路训练计算的实现任务

沿用 [TRAINING.md](TRAINING.md) 中的训练定义。每step为512个prompt、每题4条rollout，共2048条回答；每次optimizer更新512条回答，每step更新四次。学习率1e-6，完整100steps，每10steps保存和验证。数据、方法公式、reward、KL、entropy、token whitening、动态microbatch规则和原rollout seed设置按现有实现执行。

在 `vendor/verl/verl/trainer/ppo/ray_trainer.py::_balance_batch` 保留四路原始分组以及每次optimizer更新的样本归属，再把逻辑rank0/1交给物理rank0、逻辑rank2/3交给物理rank1。同一个物理rank顺序处理两路；每路每次optimizer更新仍为128条回答，名义microbatch为64、名义累积次数为2，动态token预算为16384。接线应保留四路原本各自的microbatch切分结果和同步所需的microbatch数量。

在 `vendor/verl/verl/workers/fsdp_workers.py`、`vendor/verl/verl/workers/actor/dp_actor.py`中分别处理物理FSDP规模与逻辑样本分组。设四路在原规则下的局部梯度贡献为g0、g1、g2、g3；两个物理rank分别累积(g0+g1)/2与(g2+g3)/2，然后经两rank平均得到(g0+g1+g2+g3)/4。每路仍使用原microbatch token-mean loss除以名义累积次数2，再乘物理归并系数1/2。全部贡献累积完后，执行一次原规则的全局梯度裁剪及optimizer更新。

Actor/ref log-prob计算也沿用四路的样本边界和动态microbatch分组。`vendor/verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py`保留四个逻辑rollout流对应的prompt分配、请求顺序及各自持续的随机状态；物理卡轮流执行它们。validation同样保留原分配规则。四路输入、输出、mask和原始顺序在回到trainer时正确合并。沿用当前vLLM0.6.3、XFORMERS和seed设置。

用同一份固定rollout、advantages、模型和optimizer状态，分别走原四卡与新两卡执行路径，比较每次更新的样本ID、microbatch边界、loss、裁剪前梯度、梯度范数以及更新后参数；包含不同回答长度和GD²PO-Hard的query weight。样本归属与权重必须一致；浮点误差记录绝对/相对值，并用相同硬件的重复执行误差作参照。再检查四路rollout请求分配与随机状态能独立延续，跑到step10完成validation和HF checkpoint加载。验证发现分组或权重不一致时修正接线，再切换正式任务。修改算法或训练接线后运行 `bash scripts/test.sh`，并保留GPU对照结果。

## 安装、并发隔离与输出

按 [HANDOFF.md](HANDOFF.md) 安装 `rdgdpo`、`dara-inference`、`dara-bfcl-v4` 三个环境，获取固定数据和Qwen2.5-1.5B-Instruct。训练只使用本次15个run与原seed。每个run独立保存命令、展开配置、日志、exit.code和模型；两卡运行另记录physical_world_size=2、logical_world_size=4及实际GPU型号。

原四卡路径的可执行命令如下。两卡接线完成并验证后，由执行agent把实际两卡启动命令写入本文件，并用启动器的dry-run确认配置已生效。

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

每个训练任务使用独立CUDA_VISIBLE_DEVICES与Ray临时目录；控制器为各任务分配独立端口和日志。检查本机CPU/RAM、共享内存和磁盘IO是否能同时支撑四个训练进程及checkpoint导出，按实际进程需要分配。每次100step训练有10份HF模型，15次约996GiB，准备至少2TB持久空间并额外计入BFCL原始输出。任务失败时保留该attempt的日志和模型，释放其进程占用并立即派发可执行的其他任务；训练重试使用新attempt目录。

## 训练、评测与租期衔接

每个完整run的step100先进入BFCL V4评测，再补齐step10至90。还有足够租期完成一个训练run时，空闲的两卡训练通道继续领取训练；租期不足时，用每张空闲卡各评测一份checkpoint。三路训练占六卡时，余下两卡同时评测两份checkpoint；训练全部结束后，八卡并行八份checkpoint评测。官方评分和CSV汇总在CPU上执行，推理完成的GPU立即接下一份模型。

20小时是本次租期预算。两卡完整run耗时记为T2，四卡记为T4：四路并行时15次训练约需4×T2，两路并行时约需8×T4；T2小于2×T4时，四路具有更高训练吞吐。先记录真实step耗时、rollout和更新耗时、GPU计算利用率、显存峰值、保存耗时，再用首个完整run校正ETA。若T2为3小时、4小时或5小时，四轮训练分别约12、16或20小时；这些是预算示例，实际ETA使用本机测量值，环境准备、GPU验证与BFCL评测另计。SXM4 80GB的容量与带宽参数见 [NVIDIA A100数据表](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf)。

每个run是否再开训练由剩余时间、最近同类run实测耗时及模型导出时间决定。评测逐case保存，租期结束前交回已完成及未完成的模型清单、原始输出与对应路径，之后复用相同输出目录补齐。持续控制器无需等用户逐项确认便可补位；任务归属和训练/评测输出必须可追溯。

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

交回150份checkpoint的原始推理、官方评分、逐题诊断和summary.json，以及 `checkpoint_per_model.csv`、`checkpoint_method_results.csv`、`process_final_per_model.csv`、`process_final_method_results.csv`。交回15个run的训练配置、完整metrics.jsonl、training_dynamics.csv、HF模型路径、硬件布局与两卡对照记录。GRPO、GDPO、DARA的曲线包含correctness/format reward、各通道π、权重和活跃group数量。最终记录明确训练完成数、已评分checkpoint数和剩余任务。
