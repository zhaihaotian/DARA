# 数据与实验工件

运行 `python scripts/fetch_data.py` 后获得实际使用的 `data/rlla_4k/train.parquet`（3920题）和 `test.parquet`（80题）。下载源固定为ToolRL提交 `8cee13ec0ca72f0461da372a93a6fd8140dbb840`；已直接比较确认两个文件与Haotian现有数据逐字节一致。BFCL评测数据由固定版本的官方evaluator单独读取。

## 可直接拿去画图的文件

| 文件 | 内容 |
|---|---|
| results/reference/per_model.csv | 44个模型、各seed的V3/V4最终分数 |
| results/reference/all_seeds.csv | 完整seed汇总，mean与sample SD分列 |
| results/reference/selected_seeds.csv | 事后筛选诊断，明确n与保留的seed |
| results/reference/training_dynamics.csv | 42个训练run，每step的format、correctness、length、validation、耗时、density |
| results/reference/training_status.csv | 完整step数、format首次达到0.8、最终format、累计训练step耗时 |
| results/reference/training_logs/ | 42份metrics.jsonl，保留每项已记录指标；仅把历史rdgdpo/日志前缀映射为dara/ |
| results/reference/local_artifact_locations.json | 本服务器上的原始日志与最终checkpoint位置 |
| results/rental_1p5b_20260919/training/training_dynamics.csv | 租机15个run的1515行逐step训练动态 |
| results/rental_1p5b_20260919/training/runs/ | 15份原始metrics、展开配置、命令和runtime记录 |
| results/rental_1p5b_20260919/evaluation/checkpoint_per_model.csv | 150份checkpoint逐模型BFCL V4结果 |
| results/rental_1p5b_20260919/evaluation/checkpoint_method_results.csv | 五个方法、十个step、三个seed的过程均值与样本标准差 |
| results/rental_1p5b_20260919/evaluation/process_final_per_model.csv | 本轮15份step100 final结果 |
| results/rental_1p5b_20260919/evaluation/process_final_method_results.csv | 本轮五个方法的final均值与样本标准差 |

训练曲线中`train_format`来自`critic/format_score/mean`，`train_correctness`来自`critic/correctness_score/mean`，correctness reward的范围为[-3,3]。`val_format`只在step0/10/…/100有值。CSV空单元格表示该项未记录；DARA的π使用对应的已记录指标。

历史模型ID保存在`source_model_id`列，供定位原始评测输出；显示方法名和新命令统一用DARA。所有已完成参考结果都来自原Haotian代码的两奖励训练。

## 大型工件位置

原1.5B训练目录为`/scratch.global/lian0190/RD-GDPO/results/figure1`，3B目录为`/scratch.global/lian0190/RD-GDPO-3B/20260913/haotian`。每run最终模型在`actor/global_step_100`。模型权重存放在对应训练目录中。

V3的原始1.5B批次位于`/scratch.global/lian0190/BFCL-v3-evaluation/20260913`；3B当前两组AST批次位于`/scratch.global/lian0190/BFCL-v3-evaluation/20260914/haotian_3b`。V4的44模型批次位于`/scratch.global/lian0190/BFCL-v4-evaluation/20260916/haotian_1p5b_dvao`，内含checkpoint_manifest.json、protocol.json、models/<model-id>/shards、result、score和summary.json。目录命名来自历史批次，V4这个目录实际包含两种规模及全部当前方法。

这些绝对路径用于现服务器定位。要在别处重新评分已有模型，需要额外同步相应raw/result工件；下载本Git仓库、创建环境并执行固定数据下载脚本后，即可运行新训练和评测；轻量结果表和训练曲线已在Git仓库中。复制已保存的BFCL分片时，按category合并为`evaluation/run.py`使用的`raw/<category>.jsonl`，保留case ID和完整metadata。

2026-09-19完成的租机工件位于`/lambda/nfs/haotian/dara-rental-20260917`：15次训练占约1.07TB，150份BFCL V4原始评测占约31GB。轻量、可直接画图和复算表格的数据已同步到[`results/rental_1p5b_20260919`](../results/rental_1p5b_20260919/README.md)。大型工件镜像使用私有Hugging Face数据仓库[`zhaihaotian/rd-gdpo-experiment-results`](https://huggingface.co/datasets/zhaihaotian/rd-gdpo-experiment-results/tree/main/rental_1p5b_20260919)。

`scripts/export_reference.py`可以从原combined per_model.csv与训练日志重新导出这份轻量参考包。来源参数显式传入，脚本选择Haotian实验记录。
