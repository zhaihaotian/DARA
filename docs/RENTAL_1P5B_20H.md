# 1.5B 八卡 A100 40GB 租机实验清单

这份清单用于确认新机器的任务分配。机器为8×A10040GB、租期20小时，拆成GPU0–3和GPU4–7两组，每组运行一个四卡实验。可执行清单位于 [rental_1p5b_20h.json](../configs/rental_1p5b_20h.json)，当前状态为准备交接。服务器获配后再登记任务归属并移交集群中的对应待运行任务。

## 固定设置

全部使用Haotian infra、Qwen2.5-1.5B-Instruct、G4、correctness+format两奖励、512个prompt/step、2048条回答/step、学习率1e-6、100steps。每10steps保存HF模型并验证，保留steps10/20/…/100。原数据、reward、动态microbatch、loss累积、KL、entropy与rollout seed设置沿用 [TRAINING.md](TRAINING.md)。训练环境安装使用 [HANDOFF.md](HANDOFF.md) 的 `rdgdpo` 环境流程。

## 过程比较的种子选择

使用每个方法原始五seed实验中训练Format Reward首次达到0.8的step，按从快到慢排序；未达到阈值的seed排在末尾，并列按seed编号升序排列。去掉排序首尾各一个，保留中间三个。最终比较继续汇总完整seed结果，过程比较记录实际使用的seed。

| 方法 | seed0 / 1 / 2 / 4 / 5 的首次达标step | 去掉最快seed | 去掉最慢seed | 过程重跑seed |
|---|---|---|---|---|
| GRPO | 37 / 43 / 34 / 31 / 31 | 4（与5并列，按编号处理） | 1 | 0、2、5 |
| GDPO | 18 / 19 / 75 / 16 / 65 | 4 | 2 | 0、1、5 |
| DARA | 14 / 16 / 15 / 13 / 29 | 4 | 5 | 0、1、2 |

DVAO和GD²PO-Hard的原始seed4、5尚待完成。DVAO已有seed0首次达标step60，seed1、2在100steps内未达标；Hard已有seed0、1首次达标step26、44，seed2未达标。先补齐各自seed4、5，再按上述同一规则确定过程比较的三个seed。

排名依据采用原始最终比较的五seed cohort。本轮DVAO seed0的完整过程模型作为已有工件保留；当seed0被选入过程比较时直接复用该工件，并保留其实际训练曲线。

## 首批确定的13次训练

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

先并行执行前四个补种子任务，约5小时后取得两种方法的完整五seed训练记录，再确定DVAO和Hard的过程补跑任务。复用已选seed的完整过程模型；只给缺少steps10/20/…/100的入选seed追加训练。按现有工件，DVAO还会需要1–2次过程补跑，Hard需要1–3次，具体seed由4、5的训练结果决定。

## 时间与输出

历史四卡1.5B完整训练约2.3–2.6小时，最近每10steps保存的DVAO训练端到端耗时2小时28分钟。首批13次训练以两路并行安排，约需17–18小时纯训练时间。完成全部过程补跑后，总训练数预计15–18次，约需20–23小时训练时间；环境安装、模型下载、数据回传与BFCL评测另计。20小时租期优先完成13次确定任务，按剩余租期继续安排额外过程补跑，短尾时间可用于评测。

每个run保存独立输出目录、展开配置、完整训练日志和十份HF checkpoint。GRPO、GDPO、DARA记录两路reward、π、权重和活跃group数量。用tmux维持两组任务，并在开始下一项前确认上一项exit.code为0。

按现有DVAO训练导出的文件实测，一份1.5B HF checkpoint约6.64GiB，13个run的130份模型约863GiB。全部15–18个run约需996–1194GiB存放模型，机器建议配置2TB可持久保存的空间，另留环境、日志和评测输出空间。

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

完成后的step100模型进入既定BFCL V4最终比较。过程比较评测选定seed的steps10/20/…/100，评分与汇总沿用 [EVALUATION.md](EVALUATION.md)。原始逐seed结果、完整曲线和选择依据一起交回。
