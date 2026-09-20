"""Package the 1.5B checkpoint curves, their CSV inputs, and a standalone plotter."""
import csv
import inspect
import runpy
import shutil
import zipfile
from pathlib import Path

import build_figures as source

ROOT = Path(__file__).resolve().parent
EXPORT = ROOT / 'dynamics_export'


def write_csv(name, rows):
    with (EXPORT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    EXPORT.mkdir(exist_ok=True)
    per_seed = source.curve_rows(source.read_csv(ROOT/'checkpoint_per_run.csv'))
    for row in per_seed:
        row['seed'], row['step'] = int(row['seed']), int(row['step'])
        for metric in source.METRICS:
            row[metric] = float(row[metric])
    source.audit(per_seed, ['run_id', 'step'])
    summary = source.aggregate(per_seed)
    rlla_rows = source.rlla_curve_rows(per_seed)
    rlla_stats = source.aggregate_rlla(rlla_rows)
    all_runs = [r for r in source.read_csv(ROOT/'checkpoint_per_run_all.csv') if r['model_size']=='1.5B']
    selection = [r for r in source.read_csv(ROOT/'run_selection.csv') if r['cohort']=='checkpoint' and r['model_size']=='1.5B']
    for row in selection:
        row['in_process_figure'] = int(row['seed']) in source.CURVE_SEEDS[row['method']]
    base = next(r for r in source.read_csv(ROOT/'final_per_run.csv') if r['method']=='base' and r['model_size']=='1.5B')
    write_csv('per_seed.csv', per_seed)
    write_csv('method_summary.csv', summary)
    write_csv('all_runs.csv', all_runs)
    write_csv('seed_selection.csv', selection)
    write_csv('rlla_per_seed.csv', rlla_rows)
    write_csv('rlla_method_summary.csv', rlla_stats)
    for name, rows in [('curve_per_run.csv', per_seed), ('curve_method_results.csv', summary),
                       ('rlla_per_run.csv', rlla_rows), ('rlla_method_results.csv', rlla_stats)]:
        source.save_csv(name, rows)
    plot_rows = []
    for method in source.METHODS:
        initial = dict(model_size='1.5B', method=method, step=0, n=1, seeds='')
        for metric in source.METRICS:
            initial[metric+'_mean'] = base[metric]
            initial[metric+'_sd'] = ''
        plot_rows.append(initial)
        plot_rows.extend(sorted([r for r in summary if r['method']==method], key=lambda r:int(r['step'])))
    write_csv('plot_data.csv', plot_rows)

    header = '''"""Draw BFCL V4 checkpoint curves from the accompanying plot_data.csv."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
'''
    for name in ['METHODS', 'LABELS', 'COLORS']:
        header += name + ' = ' + repr(getattr(source,name)) + '\n'
    entry = '''
def main():
    with (OUT / 'plot_data.csv').open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row['step'] = int(row['step'])
        for key in list(row):
            if key.endswith(('_mean', '_sd')) and row[key] != '':
                row[key] = float(row[key])
    initial = next(row for row in rows if row['step'] == 0)
    base = {metric: initial[metric + '_mean'] for metric in ['average_accuracy', 'average_format']}
    draw_curves([row for row in rows if row['step'] > 0], base)
    with (OUT / 'rlla_method_summary.csv').open() as f:
        rlla = list(csv.DictReader(f))
    for row in rlla:
        row['step'] = int(row['step'])
        for key in list(row):
            if key.endswith(('_mean', '_sd')) and row[key] != '':
                row[key] = float(row[key])
    draw_rlla_curves(rlla)


if __name__ == '__main__':
    main()
'''
    (EXPORT/'plot_dynamics.py').write_text(header+'\n\n'+inspect.getsource(source.save_figure)+'\n\n'+inspect.getsource(source.draw_curves)+'\n\n'+inspect.getsource(source.draw_rlla_curves)+entry)
    (EXPORT/'README.md').write_text('''# Qwen2.5-1.5B-Instruct：RLLA validation 与 BFCL V4 过程图

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
''')
    shutil.copyfile(ROOT/'FIGURE_CAPTION.txt', EXPORT/'FIGURE_CAPTION.txt')
    rlla_caption = ('Checkpoint-wise RLLA validation of Qwen2.5-1.5B-Instruct. The panels show correctness reward and format compliance. '
                    'GRPO uses seeds 0/2/5, GDPO uses 0/1, and DARA uses 0/1/2. DVAO and GD2PO-Hard show selected seeds 3 and 4 respectively, '
                    'chosen for earlier Format convergence among their completed process runs. Both panels show mean curves. Validation is recorded every 10 training steps, including step 0.\n')
    (ROOT/'RLLA_FIGURE_CAPTION.txt').write_text(rlla_caption)
    (EXPORT/'RLLA_FIGURE_CAPTION.txt').write_text(rlla_caption)
    # Execute the exact standalone script delivered to the user.
    runpy.run_path(str(EXPORT/'plot_dynamics.py'), run_name='__main__')
    for name in ['bfcl_v4_checkpoint_dynamics', 'bfcl_v4_average_accuracy', 'bfcl_v4_average_format',
                 'rlla_validation_dynamics', 'rlla_validation_correctness', 'rlla_validation_format']:
        for ext in ['png', 'pdf', 'svg']:
            shutil.copyfile(EXPORT/f'{name}.{ext}', ROOT/f'{name}.{ext}')
    with zipfile.ZipFile(ROOT/'bfcl_v4_1p5b_dynamics.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(EXPORT.iterdir()):
            if path.is_file():
                archive.write(path, arcname='bfcl_v4_1p5b_dynamics/'+path.name)
    print(f'Exported {len(per_seed)} retained run-step rows, {len(summary)} summaries, {len(plot_rows)} plotted points.')
    print(ROOT/'bfcl_v4_1p5b_dynamics.zip')


if __name__ == '__main__':
    main()
