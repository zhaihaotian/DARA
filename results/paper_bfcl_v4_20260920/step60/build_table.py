"""Export BFCL V4 step-60 results from the saved checkpoint evaluations."""
import csv
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
PARENT = OUT.parent
sys.path.insert(0, str(PARENT))
import build_figures as source
import matplotlib.pyplot as plt


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    candidates = [r for r in source.read_csv(PARENT / 'checkpoint_per_run_all.csv') if int(r['step']) == 60]
    bases = [r for r in source.read_csv(PARENT / 'final_per_run.csv') if r['method'] == 'base']
    for r in candidates + bases:
        r['step'] = int(r['step'])
        if r['seed'] != '':
            r['seed'] = int(r['seed'])
        for metric in source.METRICS:
            r[metric] = float(r[metric])
    for r in candidates:
        if r['source'].startswith('inputs/'):
            log = PARENT / 'inputs' / (r['run_id'] + '.metrics.jsonl')
        else:
            log = PARENT.parent / r['source'].split('/')[0] / 'training/runs' / r['run_id'] / 'metrics.jsonl'
        metric = next(m for m in source.read_metrics(log) if m['step'] == 60)
        r.pop('validation_format_step100')
        r.pop('training_format_step100')
        r['validation_format_step60'] = float(metric['val/test_format/rlla'])
        r['training_format_step60'] = float(metric['critic/format_score/mean'])
        r['retained'] = r['validation_format_step60'] >= source.MIN_FORMAT
        r['exclusion_reason'] = '' if r['retained'] else 'step60 RLLA validation Format < 0.8'
    runs = [r for r in candidates if r['retained']]
    # Detect duplicated run-seed rows and incorrect macro averages before export.
    source.audit(candidates + bases, ['model_size', 'method', 'seed', 'step'])
    stats = source.aggregate(runs + bases)
    lookup = {(r['model_size'], r['method']): r for r in stats}

    ordered = [(size, method, lookup.get((size, method))) for size in ['1.5B', '3B'] for method in ['base'] + source.METHODS]
    table_rows = [row for row in ordered if row[0] == '1.5B']
    best = {size: {m: max(r[m + '_mean'] for r in stats if r['model_size'] == size and r['method'] != 'base') for m in source.METRICS} for size in ['1.5B', '3B']}
    def display(row, metric, math=False):
        if row is None:
            return '—' if not math else r'\textemdash{}'
        bold = row['method'] != 'base' and round(row[metric + '_mean'], 2) == round(best[row['model_size']][metric], 2)
        return source.cell(row, metric, math=math, bold=bold)

    write_csv('per_run.csv', runs)
    write_csv('selection.csv', candidates)
    write_csv('method_results.csv', [r for r in stats if r['method'] != 'base'])
    write_csv('base_reference.csv', source.aggregate(bases))
    coverage = [dict(model_size=size, method=method, step=60, n=row['n'] if row else 0,
                     seeds=row['seeds'] if row else '', status='evaluated' if row else 'only_step100_saved')
                for size, method, row in ordered if method != 'base']
    write_csv('coverage.csv', coverage)
    (OUT / 'verification.json').write_text(json.dumps(dict(
        selection_metric='step60 val/test_format/rlla', threshold=source.MIN_FORMAT,
        candidates=len(candidates), retained=len(runs), excluded=len(candidates)-len(runs),
        group_average_max_error=source.audit(runs, ['model_size', 'method', 'seed', 'step']),
        coverage=coverage), indent=2) + '\n')

    caption = ('BFCL V4 for Qwen2.5-1.5B-Instruct at training step 60. Acc. denotes tool-calling accuracy and Format denotes output-format compliance. '
               'Average is the mean over Live, Non-Live, and Multi-Turn. Results are percentages, reported as mean $\\pm$ sample standard deviation. '
               'Base is the untrained model. Checkpoints are retained when step-60 RLLA validation Format is at least 80\\%. '
               'GRPO and DARA have three retained seeds each, GDPO has two, and DVAO and GD$^2$PO-Hard have one each. Single-seed results are shown as point estimates.')
    tex = [r'\begin{table*}[t]', r'\centering', r'\caption{' + caption + '}', r'\label{tab:bfcl-v4-step60}',
           r'\setlength{\tabcolsep}{4pt}', r'\small', r'\begin{tabular}{llrrrrrrrr}', r'\toprule',
           r' & & \multicolumn{2}{c}{Live} & \multicolumn{2}{c}{Non-Live} & \multicolumn{2}{c}{Multi-Turn} & \multicolumn{2}{c}{Average} \\',
           r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}',
           r'Size & Method & Acc. & Format & Acc. & Format & Acc. & Format & Acc. & Format \\', r'\midrule']
    md = ['# BFCL V4：1.5B step 60', '', '| Size | Method | Live Acc. | Live Format | Non-Live Acc. | Non-Live Format | Multi-Turn Acc. | Multi-Turn Format | Average Acc. | Average Format |',
          '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fig, ax = plt.subplots(figsize=(16, 5.1))
    ax.set(xlim=(-.05, 10.8), ylim=(3.5, 14.6))
    ax.axis('off')
    xcols = [2.45 + i * 1.1 for i in range(8)]
    ax.hlines(14, -.05, 10.6, lw=1.5, color='black')
    for i, group in enumerate(['Live', 'Non-Live', 'Multi-Turn', 'Average']):
        ax.text((xcols[2*i]+xcols[2*i+1])/2, 13.55, group, ha='center', va='center', fontsize=16, fontfamily='DejaVu Serif')
        ax.hlines(13.15, xcols[2*i]-.48, xcols[2*i+1]+.48, lw=.8, color='.3')
    for x, label in zip([.02, .63] + xcols, ['Size', 'Method'] + ['Acc.', 'Format'] * 4):
        ax.text(x, 12.65, label, ha='left' if x < 1 else 'center', va='center', fontsize=14, fontfamily='DejaVu Serif')
    ax.hlines(12.2, -.05, 10.6, lw=1, color='black')
    for i, (size, method, row) in enumerate(table_rows):
        label = source.LABELS[method]
        tex.append(' & '.join([size if i in [0, 6] else '', label] + [display(row, m, math=True) for m in source.METRICS]) + r' \\')
        md.append('| ' + ' | '.join([size, source.PLAIN[method]] + [display(row, m) for m in source.METRICS]) + ' |')
        y = 11.75 - i * .91
        if i in [0, 6]:
            ax.text(.02, y, size, va='center', fontsize=14, fontfamily='DejaVu Serif')
        ax.text(.63, y, label, va='center', fontsize=14, fontfamily='DejaVu Serif')
        for x, metric in zip(xcols, source.METRICS):
            value = display(row, metric, math=True) if row else '—'
            ax.text(x, y, value, ha='center', va='center', fontsize=13.7)
        if i == 5 and len(table_rows) > 6:
            tex.append(r'\midrule')
            ax.hlines(y-.455, -.05, 10.6, lw=.9, color='black')
    ax.hlines(6.75, -.05, 10.6, lw=1.5, color='black')
    notes = ['Qwen2.5-1.5B-Instruct · BFCL V4 · training step 60 · mean ± sample SD (%)',
             'GRPO / DARA n=3, GDPO n=2, DVAO / GD²PO-Hard n=1. Base: untrained reference.',
             'Retained checkpoints: step-60 RLLA validation Format ≥ 80%. Single-seed results: point estimates.',
             'Average = mean over Live, Non-Live, and Multi-Turn.']
    for i, note in enumerate(notes):
        ax.text(5.25, 6.02-i*.53, note, ha='center', fontsize=11 if i == 0 else 10)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / ('table_bfcl_v4_step60.' + ext), dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    (OUT / 'table_bfcl_v4_step60.tex').write_text('\n'.join(tex) + '\n')
    md += ['', '数值为百分数，跨 seed 计算 mean ± sample SD（ddof=1）。Average 先对每个 run 的 Live、Non-Live、Multi-Turn 取算术平均，再跨 run 汇总。Base 为训练前模型。', '',
           '1.5B：GRPO seeds 0/2/5，GDPO 0/1，DVAO 3，GD²PO-Hard 4，DARA 0/1/2。3B：DVAO 与 GD²PO-Hard 均为 seeds 0/1/2。DVAO 与 GD²PO-Hard 的 1.5B 结果各来自一个 seed，显示单次分数。', '',
           '纳入条件为当前 checkpoint 的收敛状态：step60 RLLA validation Format ≥0.8，同样应用到所有训练方法。读取每个 run 的 step60 训练日志后筛选，共21个候选checkpoint，保留16个。selection.csv 保存逐 checkpoint 的日志指标、纳入状态及排除原因。', '',
           '图表展示1.5B的Base及五个方法。CSV保留1.5B和3B全部现有step60数据，可按model_size筛选。', '',
           '本表来自最新 PR #1（a954110）的过程评测，以及本地已完成的 3B DVAO / GD²PO-Hard seed0 过程评测。上级 inputs 目录保留本地数据快照。per_run.csv 为逐 run 结果，method_results.csv 为统计表，coverage.csv 为各方法 seed 覆盖情况。', '',
           '生成命令：`python results/paper_bfcl_v4_20260920/step60/build_table.py`。脚本依赖上级 build_figures.py、CSV、NumPy 和 Matplotlib。LaTeX 使用 booktabs。']
    (OUT / 'README.md').write_text('\n'.join(md) + '\n')
    print('Exported step60 table:', len(runs), 'evaluations;', len(stats)-2, 'method/size groups.')
    for r in stats:
        if r['method'] != 'base':
            print(r['model_size'], source.PLAIN[r['method']], 'n='+str(r['n']), 'Average Acc.='+source.cell(r, 'average_accuracy'), 'Average Format='+source.cell(r, 'average_format'))


if __name__ == '__main__':
    main()
