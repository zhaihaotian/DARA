# 最终比较与过程 checkpoint 实验

本轮使用 Haotian infra、Qwen2.5-Instruct、G4、correctness+format 两奖励、100训练steps和现有固定数据。任务定义见 [checkpoint_study.json](../configs/checkpoint_study.json)。

## 最终比较

最终比较评测step100的BFCL V4。GRPO、GDPO、DARA在1.5B和3B各保留五seed的历史训练与最终评测。1.5B的DVAO、GD²PO-Hard已有seeds0/1/2，各新增4/5；3B的DVAO、GD²PO-Hard各运行seeds0/1/2，共新增10次最终训练。RVPO与GDPO-SAW复用已有结果。

新增训练统一每10steps保存模型。任务顺序优先覆盖3B的seeds0/1/2，再补1.5B的4/5；每次分配选择剩余租期能够覆盖的最前任务。

## 1.5B过程比较

1.5B的上述五个方法各保留三个seed，逐个评测steps10/20/…/100。GRPO选择0/2/5，GDPO选择0/1/5，DARA选择0/1/2；这三个方法按原始五seed训练Format Reward首次达到0.8的step升序排列，相同step按seed升序，去掉首尾各一个，保留中间三个。原始逐seed数据与选择依据保存在manifest。该选择规则随过程比较结果一起披露。

DVAO的过程种子为3/4/5，新增seed3过程训练，并复用新增seed4/5的过程模型。GD²PO-Hard的过程种子为0/4/5，seed0在原始训练中首次达到Format0.8为step26；重跑seed0并复用新增4/5。除10个最终训练任务外，另需11次1.5B过程训练：GRPO、GDPO、DARA各三次，DVAO seed3和GD²PO-Hard seed0各一次。两个尺寸的最终比较优先获得训练资源，后续可用资源接入过程训练。

1.5B租机为8×A100 80GB SXM4，包含15个待运行任务，清单见 [RENTAL_1P5B_20H.md](RENTAL_1P5B_20H.md)，执行agent使用独立的 [RENTAL_A100_80G_SXM4_AGENT.md](RENTAL_A100_80G_SXM4_AGENT.md)。机器获配后登记任务归属，再将对应任务从集群派发移交。

这批共21个训练任务。1.5B过程评测共150个checkpoint；3B新增训练评测6个step100模型，合计156个新增BFCL V4模型评测。

## 启动与保存

在训练环境中使用manifest启动一个待执行任务：

```bash
conda activate rdgdpo
python training/run_manifest.py --manifest configs/checkpoint_study.json \
  --id haotian_1p5b_dvao_s4_save10 \
  --model-root /shared/models --output-root outputs
```

每次训练放在tmux中运行，输出使用独立目录。当前正式自动训练使用四个物理rank，4张A100或H100运行一个run，8张卡运行两个run。两份同型两卡allocation可以拼接，启动方式见 [MULTINODE.md](MULTINODE.md)。单独两卡资源用于评测。

后续两卡训练需要保留原四路逻辑分组，让每张物理卡顺序执行两路，保持原来的每次optimizer更新样本、动态microbatch边界、loss权重和rollout请求分配。验收使用固定rollout、advantages和optimizer状态，对比loss、裁剪前梯度及参数更新。完整接线与验收尚待完成；全局batch参数相同不足以证明两卡与四卡的梯度口径相同。

3B DVAO seed0在2026-09-17使用两张H100、两个物理rank完成训练，用户要求保留这一次实验和结果。它继续进入既定最终评测，硬件配置作为该run的记录保留。其他固定参数见 [TRAINING.md](TRAINING.md)。

模型与tokenizer保存在 `actor/global_step_10` 至 `actor/global_step_100`。训练日志逐step记录reward；GRPO、GDPO、DARA还记录各通道π、权重、活跃group数量。验证在steps0/10/…/100执行。保存的是用于推理的HF actor模型；租期中断的训练以新attempt目录从相同base与seed重新开始，原attempt的模型和日志保留。

## BFCL V4与表格

每次训练完整完成后，检查该次attempt的目标模型文件齐全，再送入相同的BFCL V4队列。step100优先评测；过程checkpoint按训练step排列。调用 [evaluation/run.py](../evaluation/run.py) 时，把 `--model` 指向所需step的HF模型目录即可，其余评测参数保持 [EVALUATION.md](EVALUATION.md) 的设置。

使用14类、3301cases，Non-Live AST按四组macro平均，Simple中的Python/Java/JavaScript等权；Live AST按1351cases加权；Multi-Turn为四类各200cases的均值。Average为三大组等权平均。Format按每题assistant输出的RLLA结构符合率平均，再沿用相同类别权重。各checkpoint先独立算分，然后在相同模型尺寸、方法、step内计算跨seed均值与样本标准差。

历史最终比较汇总44个已完成历史模型和10个新增seed的最终模型，共54个模型。本轮15次1.5B训练还单独汇总各自的step100最终结果，包括租机清单第15项DVAO seed3。step100的同一份评分同时用于过程曲线终点和本轮最终结果。每条原始结果保留具体模型路径、step、seed和训练attempt。

MSI当前执行目录为 `/scratch.global/lian0190/RD-GDPO-3B/20260913`，自动训练记录为 `runs.json`，任务文件为 `dara_training_tasks.json`。新训练进入 `dara_save10/<run-id>/attempt_<n>`。已完成的DVAO seed0训练和过程模型保留在 `/scratch.global/lian0190/DARA/20260917/1p5b-dvao-g4-s0-two-save10`，复跑对照见 [DVAO_SEED0_REPLAY.md](DVAO_SEED0_REPLAY.md)。

新评测目录为 `/scratch.global/lian0190/BFCL-v4-evaluation/20260917/dara_checkpoint_study`。`report/final_per_model.csv` 与 `final_method_results.csv`给出历史最终比较及新增seed结果；`checkpoint_per_model.csv`与`checkpoint_method_results.csv`给出过程结果；`process_final_per_model.csv`与`process_final_method_results.csv`给出本轮15次1.5B训练的step100最终结果。原始回答、官方评分和Format统计保存在各 `models/<model-id>` 目录。训练调度、评测派发和跨seed汇总由登录节点tmux内的持久控制器执行，每分钟检查一次资源和任务状态，每30分钟保存一次进度报告。
