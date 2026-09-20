# BFCL V4 final evaluation 与 checkpoint dynamics

本次整理读取 DARA PR #1 的 `a954110` 提交、仓库 `results/reference` 中的历史最终结果，以及本地已经完成的 3B DVAO / GD²PO-Hard seed0 过程评测。所有表格和曲线使用已保存的评测数据。最终表对应用户提供的 Table 1，过程图对应 Qwen2.5-1.5B-Instruct 的五方法比较。

## Table 1

[`table1_bfcl_v4.pdf`](table1_bfcl_v4.pdf)、[`table1_bfcl_v4.png`](table1_bfcl_v4.png) 给出完整的 1.5B / 3B 表，LaTeX 为 [`table1_bfcl_v4.tex`](table1_bfcl_v4.tex)，文本表为 [`TABLE1.md`](TABLE1.md)，逐 run 数据与汇总分别为 `final_per_run.csv`、`final_method_results.csv`。LaTeX 使用 `booktabs`，每列的最高显示均值加粗。

Base、GRPO、GDPO、DARA沿用原 Table 1 的结果；GRPO、GDPO、DARA每种尺寸均保留历史 seeds0/1/2/4/5。1.5B DVAO、GD²PO-Hard在原始 seeds0/1/2上补入新的4/5。3B DVAO、GD²PO-Hard汇总本地seed0和PR新增seed1/2。

本次按用户要求筛选未收敛run，使用每个run在step100的RLLA validation Format ≥0.8作为纳入条件。该条件一致应用到全部训练方法。最终表中，1.5B DVAO保留seeds0/2/4，排除1/5；1.5B GD²PO-Hard保留0/1/4，排除2/5。3B DVAO和GD²PO-Hard的0/1/2均满足条件。每个方法每个seed在最终表计入一次。

1.5B DVAO更新后的Average Accuracy为48.64±2.64%，Average Format为93.52±2.91%；GD²PO-Hard分别为50.70±0.98%、96.35±0.73%。DARA分别为51.17±0.78%、97.55±0.64%，两项均是本表1.5B比较方法中的最高均值。

3B DVAO的Average Accuracy为54.22±0.24%，Average Format为97.04±0.47%；GD²PO-Hard为53.90±1.11%、97.00±0.82%。DARA为54.15±0.58%、97.22±0.66%。DARA相比GDPO的Accuracy提高0.49个百分点、Format提高0.65个百分点；DVAO的Accuracy均值比DARA高约0.07个百分点，DARA的Format均值最高。

用户截图中旧版1.5B GD²PO-Hard三组Accuracy为68.84、79.94、5.00，其算术平均应为51.26；本次表格重新从逐run数据计算所有Average，新补seed后的GD²PO-Hard结果为50.70。

## 1.5B 过程图

[`bfcl_v4_checkpoint_dynamics.pdf`](bfcl_v4_checkpoint_dynamics.pdf)、[`bfcl_v4_checkpoint_dynamics.png`](bfcl_v4_checkpoint_dynamics.png) 是双面板图。左图为Average Accuracy，右图为Average Format。独立面板是 [`bfcl_v4_average_accuracy.pdf`](bfcl_v4_average_accuracy.pdf) 与 [`bfcl_v4_average_format.pdf`](bfcl_v4_average_format.pdf)，同时输出PNG和SVG。英文图注在 [`FIGURE_CAPTION.txt`](FIGURE_CAPTION.txt)。

五个方法使用租机过程重跑结果。GRPO为seeds0/2/5，DARA为0/1/2，GDPO为0/1，DVAO固定为seed3，GD²PO-Hard固定为seed4。DVAO和Hard各选取一条更早Format收敛的完整轨迹，n=1；其余方法对指定seeds取均值。纳入run的steps10/20/…/100全部绘制，每条曲线的seed集合固定。step0复用既有1.5B base的BFCL V4结果。

Accuracy和Format两个面板均展示均值曲线，原始评测点之间用直线连接。绘图的100条逐run数据位于 `curve_per_run.csv`，50条方法汇总位于 `curve_method_results.csv`。这两个文件包含所有八项分组指标，单seed的标准差留空。完整过程数据保存在 `checkpoint_per_run_all.csv`，按最终收敛筛选的多seed数据保存在 `checkpoint_per_run.csv`。可独立运行的绘图脚本、两套过程图和CSV位于 `dynamics_export/`，打包文件为 `bfcl_v4_1p5b_dynamics.zip`。

RLLA validation图为 `rlla_validation_dynamics.png/pdf/svg`，左右面板分别展示 `val/test_correctness/rlla` 与乘100后的 `val/test_format/rlla`。两套图使用完全相同的run集合。`rlla_per_run.csv`包含110条逐seed记录，`rlla_method_results.csv`包含55条方法汇总，覆盖steps0/10/…/100。RLLA的step0来自各run的训练前验证日志。英文图注为 `RLLA_FIGURE_CAPTION.txt`。

BFCL V4的Format优势出现得很早：step10，DARA的Average Format为73.14%，GDPO为1.40%；step20分别为78.15%、42.74%。按均值曲线首次达到阈值的已评测checkpoint，80%对应DARA step30、GDPO step40；90%对应step40、step70；95%对应step60、step100。这些时间的分辨率为10个training steps。

Accuracy曲线中，DARA在step10、20低于GDPO，step30–80高于GDPO，step90略低，step100分别为51.32%与51.15%。这组数据支持更快获得高Format、训练中后段保持有竞争力的下游Accuracy。`process_final_per_run.csv` 与 `process_final_method_results.csv` 按原多seed口径汇总过程run的step100：DVAO使用3/4，Hard使用0/4，各n=2。历史final表仍使用 `final_per_run.csv` 与 `final_method_results.csv`。绘图的单seed选择仅作用于过程图。

1.5B DVAO seed4在训练step99才首次达到training Format0.8，step100 validation Format达到0.9625，其最终Average Accuracy为45.59%。它按最终收敛条件计入final多seed汇总。

## 3B 过程结果

3B DVAO与GD²PO-Hard各有seeds0/1/2的完整steps10/20/…/100评测，共60个checkpoint。全部已经汇入 `checkpoint_per_run.csv` 和 `checkpoint_method_results.csv`，用 `model_size=3B` 筛选即可。seed0的评测和训练日志快照保存在本目录 `inputs/`，seed1/2来源于 `results/rental_3b_20260920`。两种方法的step100同时用于最终表。

## 指标与计算

每个checkpoint覆盖14类、3301个BFCL V4 cases，使用仓库固定的ToolRL prompt/parser、BFCL官方Accuracy checker和RLLA Format检查函数。推理temperature0.6、top-p0.95、seed0、最大8192个新tokens。各组及Format的具体定义见 [`docs/EVALUATION.md`](../../docs/EVALUATION.md)。

每个run先计算Live、Non-Live、Multi-Turn的Accuracy和Format，Average为这三组的算术平均。随后跨纳入run计算mean±sample SD，标准差使用ddof=1。Average的标准差直接来自各run的Average值。Base使用一次评测。

完整run与筛选记录保存在 `final_per_run_all.csv`、`checkpoint_per_run_all.csv`、`run_selection.csv`、`excluded_runs.csv`。对应的完整run均值表为 `final_method_results_all_runs.csv` 与 `checkpoint_method_results_all_runs.csv`。最终表48个候选模型含两个base，保留44个；过程评测合计210个checkpoint，保留180个，其中1.5B120个、3B60个。

本次检查覆盖逐run三个分组与Average的算术一致性、最终表唯一seed、过程唯一run-step和每run十个checkpoint的完整性。`verification.json` 记录检查结果。源码和CSV可直接重新生成图表：

```bash
python results/paper_bfcl_v4_20260920/build_figures.py
```

脚本需要Python、NumPy和Matplotlib。数据路径相对仓库解析；1.5B与3B的PR输入文件保留在各自 `results/rental_*` 目录。
