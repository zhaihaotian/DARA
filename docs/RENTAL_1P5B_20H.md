# 1.5B 八卡 A100 40GB 租机实验清单

这份清单用于确认新机器的任务分配。机器为8×A10040GB、租期20小时，拆成GPU0–3和GPU4–7两组，每组运行一个四卡实验。可执行清单位于 [rental_1p5b_20h.json](../configs/rental_1p5b_20h.json)，当前状态为准备交接。服务器获配后再登记任务归属并移交集群中的对应待运行任务。

## 固定设置

全部使用Haotian infra、Qwen2.5-1.5B-Instruct、G4、correctness+format两奖励、512个prompt/step、2048条回答/step、学习率1e-6、100steps。每10steps保存HF模型并验证，保留steps10/20/…/100。原数据、reward、动态microbatch、loss累积、KL、entropy与rollout seed设置沿用 [TRAINING.md](TRAINING.md)。训练环境安装使用 [HANDOFF.md](HANDOFF.md) 的 `rdgdpo` 环境流程。

## 过程比较的种子选择

GRPO、GDPO、DARA使用原始五seed实验中训练Format Reward首次达到0.8的step，按从快到慢排序；未达到阈值的seed排在末尾，并列按seed编号升序排列。去掉排序首尾各一个，保留中间三个。最终比较继续汇总完整seed结果，过程比较记录实际使用的seed。

| 方法 | seed0 / 1 / 2 / 4 / 5 的首次达标step | 去掉最快seed | 去掉最慢seed | 过程重跑seed |
|---|---|---|---|---|
| GRPO | 37 / 43 / 34 / 31 / 31 | 4（与5并列，按编号处理） | 1 | 0、2、5 |
| GDPO | 18 / 19 / 75 / 16 / 65 | 4 | 2 | 0、1、5 |
| DARA | 14 / 16 / 15 / 13 / 29 | 4 | 5 | 0、1、2 |

DVAO使用seed3、4、5做过程比较，新增seed3过程训练，同时补齐seed4、5的最终训练并保存全部过程模型。GD²PO-Hard使用seed0、4、5；原始seed0首次达到Format0.8为step26，重跑该seed并补齐seed4、5。

DVAO seed3完成steps10/20/…/100的BFCL V4过程评测，同时交付step100最终评测结果。本轮最终结果表覆盖全部15次训练，DVAO汇总seed3/4/5；历史五seed最终比较沿用原始seed0/1/2及新增4/5。已完成的DVAO seed0重跑日志、模型和评测结果单独保留，复跑对照见 [DVAO_SEED0_REPLAY.md](DVAO_SEED0_REPLAY.md)。

## 需要执行的15次训练

| 顺序 | 方法 | Seed | 目的 | Run ID |
|---:|---|---:|---|---|
| 1 | DVAO | 4 | 补齐最终比较，保存过程模型 | `haotian_1p5b_dvao_s4_save10` |
| 2 | GD²PO-Hard | 4 | 补齐最终比较，保存过程模型 | `haotian_1p5b_gd2po_hard_s4_save10` |
| 3 | DVAO | 5 | 补齐最终比较，保存过程模型 | `haotian_1p5b_dvao_s5_save10` |
| 4 | GD²PO-Hard | 5 | 补齐最终比较，保存过程模型 | `haotian_1p5b_gd2po_hard_s5_save10` |
| 5 | GRPO | 0 | 过程checkpoint比较 | `haotian_1p5b_grpo_s0_save10` |
| 6 | GDPO | 0 | 过程checkpoint比较 | `haotian_1p5b_gdpo_s0_save10` |
| 7 | DARA | 0 | 过程checkpoint比较 | `haotian_1p5b_dara_s0_save10` |
| 8 | GRPO | 2 | 过程checkpoint比较 | `haotian_1p5b_grpo_s2_save10` |
| 9 | GDPO | 1 | 过程checkpoint比较 | `haotian_1p5b_gdpo_s1_save10` |
| 10 | DARA | 1 | 过程checkpoint比较 | `haotian_1p5b_dara_s1_save10` |
| 11 | GRPO | 5 | 过程checkpoint比较 | `haotian_1p5b_grpo_s5_save10` |
| 12 | GDPO | 5 | 过程checkpoint比较 | `haotian_1p5b_gdpo_s5_save10` |
| 13 | DARA | 2 | 过程checkpoint比较 | `haotian_1p5b_dara_s2_save10` |
| 14 | GD²PO-Hard | 0 | 历史收敛seed的过程checkpoint重跑 | `haotian_1p5b_gd2po_hard_s0_save10` |
| 15 | DVAO | 3 | 过程checkpoint训练、逐step评测及step100最终评测 | `haotian_1p5b_dvao_s3_save10` |

先并行执行前四个补种子任务，再运行其余十一个过程训练。两组四卡分别承担八次和七次训练。完成15次训练后，五个方法各有三个seed的完整过程checkpoint，同时补齐1.5B最终比较。

## 时间与输出

历史四卡1.5B完整训练约2.3–2.6小时，最近每10steps保存的DVAO训练端到端耗时2小时28分钟。15次训练以两路并行安排，预计约19–21小时训练时间，环境准备、数据回传和BFCL评测另计。20小时租期按剩余时间安排完整run，最终与过程BFCL评测按已完成模型逐项接入既定队列。

每个run保存独立输出目录、展开配置、完整训练日志和十份HF checkpoint。GRPO、GDPO、DARA记录两路reward、π、权重和活跃group数量。用tmux维持两组任务，并在开始下一项前确认上一项exit.code为0。

按现有DVAO训练导出的文件实测，一份1.5B HF checkpoint约6.64GiB，15个run的150份模型约996GiB。机器建议配置2TB可持久保存的空间，另留环境、日志和评测输出空间。

以下命令在已安装环境与模型的机器上执行，另一组相应设置 `CUDA_VISIBLE_DEVICES=4,5,6,7`，使用不同run ID与日志文件：

```bash
conda activate rdgdpo
export DARA_MODELS=/path/to/dara-models
mkdir -p outputs/rental-1p5b/logs
CUDA_VISIBLE_DEVICES=0,1,2,3 python training/run_manifest.py \
  --manifest configs/rental_1p5b_20h.json \
  --id haotian_1p5b_dvao_s4_save10 \
  --model-root "$DARA_MODELS" \
  --output-root outputs/rental-1p5b \
  > outputs/rental-1p5b/logs/haotian_1p5b_dvao_s4_save10.log 2>&1
```

## 训练完成后的BFCL V4评测

清单中全部15次训练都要完成steps10/20/…/100的BFCL V4评测，共150份checkpoint结果。每份使用相同的14类3301cases、ToolRL prompt和解析器，以及temperature0.6、top_p0.95、推理seed0、最大生成8192tokens。各模型报告Non-Live AST、Live AST、Multi-Turn、Average的Accuracy与Format，共八项指标；计算定义见 [EVALUATION.md](EVALUATION.md)。每个方法在同一个训练step内汇总三个training seeds的mean±sample SD。

第15项DVAO seed3也先评测step100，交付它的最终八项指标，然后补齐其余九个过程checkpoint。step100的同一份推理与评分同时用于过程曲线终点和最终结果表。每个run完成全部十个checkpoint的评分后，才算完成该run的评测任务。

安装 [HANDOFF.md](HANDOFF.md#评测最终模型) 中的推理和V4环境后，对清单中每个run执行以下命令。每个空闲GPU运行一个checkpoint评测，使用独立端口、日志和输出目录；八卡可并行八份checkpoint评测。训练仍以四卡一组执行，评测使用已空出的GPU。

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
```

每份checkpoint保存raw逐题输出、官方评分、逐题诊断和summary.json。评测期间和全部完成后均可运行以下CPU汇总命令，生成过程曲线数据及本轮最终结果表：

```bash
python evaluation/checkpoint_results.py \
  --manifest configs/rental_1p5b_20h.json \
  --evaluation-root outputs/eval-v4-checkpoints \
  --output outputs/eval-v4-checkpoints/report
```

`checkpoint_per_model.csv`保存150份逐seed、逐step结果，`checkpoint_method_results.csv`保存五个方法各十个step的均值和样本标准差。`process_final_per_model.csv`保存本轮全部15个step100结果，包括第15项DVAO seed3；`process_final_method_results.csv`保存对应五个方法的三seed最终汇总。评分未齐时，终端报告完成数量，汇总表记录completed_seeds和expected_seeds。交回这四份CSV、原始逐题结果、训练完整曲线和种子选择依据。
