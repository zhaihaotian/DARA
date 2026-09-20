# Qwen2.5-1.5B-Instruct：RLLA validation 与 BFCL V4 过程图

解压后安装依赖并运行：

```bash
python -m pip install numpy matplotlib
python plot_dynamics.py
```

脚本读取同目录的plot_data.csv和rlla_method_summary.csv，生成BFCL双面板图bfcl_v4_checkpoint_dynamics和RLLA双面板图rlla_validation_dynamics，并分别导出独立面板。每张图输出PNG、PDF和SVG。两个面板均为均值曲线，各checkpoint之间直线连接。

plot_data.csv包含五个方法各11个点，共55行；step0复用一次未训练Base评测，之后是steps10/20/…/100。横轴使用step，纵轴使用average_accuracy_mean或average_format_mean。所有评测数值均为百分数。step0的sd字段留空。

method_summary.csv是50行BFCL方法×checkpoint汇总，包括全部八项分组指标的mean与sd；per_seed.csv是实际纳入绘图的100行逐seed结果。all_runs.csv保留租机提交的150行完整结果，seed_selection.csv列出15个run的最终收敛状态和in_process_figure绘图选取状态。标准差为样本标准差，ddof=1；Average先在每个run内对Live、Non-Live、Multi-Turn取算术平均，再跨seed汇总。单seed的sd字段留空。

两套过程图统一使用GRPO seeds0/2/5、GDPO seeds0/1、DARA seeds0/1/2、DVAO seed3和GD²PO-Hard seed4。DVAO与Hard各固定选取一条更早Format收敛的完整轨迹，n=1；其余方法对列出的seeds取均值。这些训练均来自PR的租机1.5B过程实验。Final evaluation使用既定多seed汇总。

rlla_per_seed.csv含10个run各steps0/10/…/100，共110行。correctness读取val/test_correctness/rlla，format读取val/test_format/rlla并乘100；rlla_method_summary.csv含55行均值与样本标准差。RLLA的step0使用各run实际记录的训练前验证结果。

各项标准差保存在CSV中。英文图注保存在FIGURE_CAPTION.txt和RLLA_FIGURE_CAPTION.txt。
