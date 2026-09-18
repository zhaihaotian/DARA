# 两张H100执行四路逻辑分组

每张H100顺序执行两路原始逻辑分组，两个物理rank通过FSDP通信。逻辑rank0/1放在物理rank0，逻辑rank2/3放在物理rank1。训练仍为512prompts、每题4条rollout、2048responses/step、每step四次optimizer更新；学习率、loss定义、梯度裁剪和100step目标使用现有配置，每10steps保存模型。

Trainer按四路平衡样本，保持各次optimizer更新的样本归属。Worker按四路换算mini/microbatch大小，每路每次更新为128/64条response。动态microbatch数量取四路token预算计算结果的全局最大值，各路分别调用原长度平衡函数。每个microbatch的loss除以原累积次数2，再除以本卡逻辑路数2；FSDP对两个物理rank求均值，得到原四路的平均贡献。梯度裁剪和optimizer更新在所有贡献累积完成后执行。

生成使用每张卡上的一个TP1 vLLM engine，依次处理对应两路prompt。每路保留独立的随机状态以及request/sequence计数器，CUDA生成随机流按原sharding manager规则初始化为1000+逻辑rank。训练和验证共用各路持续推进的生成状态。验证padding按四路对齐，log-prob计算沿用各逻辑分组的microbatch划分。

## 启动

在获配的两张H100内，先运行固定batch对照：

```bash
conda activate rdgdpo
python training/check_logical_dp.py --world-size 2 --device cuda \
  --reference training/reference/four_rank_ppo.json \
  --output outputs/two_gpu_validation.json
```

通过后使用原训练入口，加上 `--gpus 2`：

```bash
python training/launch.py --method gd2po_hard --seed 0 --model-size 3b \
  --model /shared/models/Qwen2.5-3B-Instruct --group-size 4 --rewards two \
  --save-freq 10 --gpus 2 --output outputs/3b-hard-s0
```

MSI自动调度已将以上两步串联，每个run先保存 `two_gpu_validation.log/json`，GPU对照成功后进入 `train.log`。每次attempt使用独立输出目录，运行物理卡数和逻辑路数写入调度记录。

## 已完成验证

固定128条回答，使用不同长度、正负优势和GD²PO-Hard query权重，调用实际actor的PPO更新代码。四rank和两rank分别通过Gloo完成四次AdamW更新。逐逻辑rank的microbatch样本与顺序完全一致；loss最大绝对差7.45e-9，裁剪前梯度2.98e-8，梯度范数2.98e-8，更新后参数1.16e-10。四rank原始结果保存在 `training/reference/four_rank_ppo.json`。

CUDA对照使用相同固定数据和PPO代码，并实际启用NCCL与FSDP，对照上述四rank结果。2026-09-18在e9的两张H100上通过：loss最大绝对差1.86e-8、梯度与范数2.98e-8、更新后参数9.31e-10，microbatch归属与顺序一致。结果保存在 `/scratch.global/lian0190/DARA/audits/20260918_logical_four/two_h100_gpu.json`。38项仓库测试通过。该对照验证固定数据下的更新计算；完整生成轨迹与训练速度由实际GPU训练记录确认。

CPU复跑命令：

```bash
python training/check_logical_dp.py --world-size 4 --device cpu --output outputs/four.json
python training/check_logical_dp.py --world-size 2 --device cpu --output outputs/two.json \
  --reference outputs/four.json
```
