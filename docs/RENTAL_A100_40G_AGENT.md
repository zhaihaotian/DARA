# 8×A100 40GB：执行 agent 交接

本次机器为单节点8×NVIDIA A100 40GB，租期20小时。每个实验独占四张GPU，同时运行两个实验。任务是完成 [15次1.5B训练清单](RENTAL_1P5B_20H.md)，保存全部过程模型，并完成150份checkpoint的BFCL V4评测及15份step100最终结果。任务清单为 [rental_1p5b_20h.json](../configs/rental_1p5b_20h.json)。这15项已登记交给租机执行agent，本地集群自动训练待办已移出对应任务。

**第一调度规则：任何时候都要最大化利用所有已租GPU。有可执行任务时，空出的GPU立即接训练或checkpoint评测。每条训练通道独立补位。** 在tmux中持续维护任务，每分钟检查任务退出、可用GPU、剩余租期及待评测模型，每30分钟记录进展与ETA。

## 四卡一组，两路训练

GPU0/1/2/3为训练通道A，GPU4/5/6/7为训练通道B。每个实验使用四个FSDP rank和四路TP1 rollout。首先在A运行DVAO seed4，在B运行GD²PO-Hard seed4；任一通道结束即领取清单中的下一个任务，优先完成DVAO seed5、GD²PO-Hard seed5，随后按清单顺序执行其余run。15次训练大致分配8次和7次，根据实际完成时间动态补位。

使用现有四卡训练入口和 [TRAINING.md](TRAINING.md) 的参数：512个prompt/step、G4、2048条回答/step、每次optimizer更新512条回答、每step更新四次、学习率1e-6、100steps、每10steps保存和验证。数据、reward、方法公式、动态microbatch、loss累积、KL、entropy、token whitening、rollout seed及memory0.6均沿用当前设置。

## 环境、启动与输出

按 [HANDOFF.md](HANDOFF.md) 安装 `rdgdpo`、`dara-inference`、`dara-bfcl-v4` 三个环境，获取固定数据和Qwen2.5-1.5B-Instruct。每个run单独记录命令、展开配置、日志、exit.code及模型路径。记录world_size=4和实际GPU列表，两个任务使用各自的Ray临时目录、端口与输出目录。

在tmux中启动通道A的首个任务：

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

另一个tmux窗口用 `CUDA_VISIBLE_DEVICES=4,5,6,7` 启动 `haotian_1p5b_gd2po_hard_s4_save10`，使用对应的独立日志文件。后续由持续派发任务的控制器逐项补位，记录完成状态及正在使用的GPU组。

每次训练保留steps10/20/…/100的十份HF模型和tokenizer，位于 `outputs/rental-1p5b/<run-id>/actor/global_step_<step>`。全部150份模型约996GiB，准备至少2TB持久空间，另计环境、训练日志及BFCL原始输出。任务失败时保留该attempt记录，清理其进程占用并立即补入可执行工作；训练重试使用新attempt目录。

## 租期与评测衔接

历史四卡1.5B完整训练约2.3–2.6小时，最近每10steps保存的DVAO训练端到端为2小时28分钟。两路完成15次训练预计约19–21小时；环境准备、数据回传和BFCL评测另计。使用新机器上实际step、validation及保存耗时更新ETA。20小时租期内，每次开新run前以剩余时间和最近同类完整run耗时判断是否来得及完成。

四卡通道有足够租期完成训练时立即补入下一项；剩余租期较短或该通道训练已完成时，四张空闲卡分别评测四份checkpoint。全部训练结束后，八卡并行评测八份checkpoint。每份评测独占一张GPU，使用独立端口和输出目录，vLLM memory0.85及既定解码设置。每个完整run先评测step100，再补齐step10至90。

官方评分与CSV汇总使用CPU，GPU推理结束后及时回收资源并补位。评测逐case保存，租期结束前交回已完成、正在运行及未完成的模型清单和路径，后续从相同评测输出目录补齐。15次训练、150份过程评分及15份最终结果的完成数量分别汇报。

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

交回150份checkpoint的原始推理、官方评分、逐题诊断和summary.json，以及 `checkpoint_per_model.csv`、`checkpoint_method_results.csv`、`process_final_per_model.csv`、`process_final_method_results.csv`。交回15个run的训练配置、完整metrics.jsonl、training_dynamics.csv、HF模型路径、硬件布局和任务运行记录。GRPO、GDPO、DARA的曲线包含correctness/format reward、各通道π、权重和活跃group数量。最终记录明确训练完成数、已评分checkpoint数和剩余任务。
