"""Build the final BFCL V4 table and checkpoint figures from saved evaluations."""
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
RESULTS = OUT.parent
METHODS = ['grpo', 'gdpo', 'dvao', 'gd2po_hard', 'dara']
LABELS = {'base': 'Base', 'grpo': 'GRPO', 'gdpo': 'GDPO', 'dvao': 'DVAO',
          'gd2po_hard': r'GD$^2$PO-Hard', 'dara': 'DARA'}
PLAIN = {**LABELS, 'gd2po_hard': 'GD²PO-Hard'}
COLORS = dict(zip(METHODS, ['#888888', '#278e2b', '#f18420', '#9665c9', '#2563df']))
GROUPS = ['live', 'non_live', 'multi_turn', 'average']
METRICS = [f'{g}_{m}' for g in GROUPS for m in ['accuracy', 'format']]
MIN_FORMAT = .8
PR_COMMIT = 'a954110'
CURVE_SEEDS = {'grpo': [0, 2, 5], 'gdpo': [0, 1], 'dvao': [3], 'gd2po_hard': [4], 'dara': [0, 1, 2]}


def curve_rows(rows):
    return [r for r in rows if r['model_size'] == '1.5B' and int(r['seed']) in CURVE_SEEDS[r['method']]]


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def save_csv(name, rows):
    path = OUT / name
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_metrics(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def finish_row(row, metrics, source):
    final = next(x for x in metrics if x['step'] == 100)
    row.update(source=source, validation_format_step100=final['val/test_format/rlla'],
               training_format_step100=final['critic/format_score/mean'])
    row['retained'] = row['validation_format_step100'] >= MIN_FORMAT
    row['exclusion_reason'] = '' if row['retained'] else 'step100 RLLA validation Format < 0.8'
    return row


def rental_rows(folder):
    root = RESULTS / folder
    rows = []
    for raw in read_csv(root / 'evaluation/checkpoint_per_model.csv'):
        row = dict(run_id=raw['run_id'], model_size=raw['model_size'].upper(),
                   method=raw['method'], seed=int(raw['seed']), step=int(raw['step']),
                   **{m: float(raw[m]) for m in METRICS})
        rows.append(finish_row(row, read_metrics(root / 'training/runs' / row['run_id'] / 'metrics.jsonl'),
                               f'{folder}/evaluation/checkpoint_per_model.csv'))
    return rows


def historical_rows():
    rows = []
    for raw in read_csv(RESULTS / 'reference/per_model.csv'):
        if raw['method'] not in ['base'] + METHODS:
            continue
        row = dict(run_id=raw['run_id'], model_size=raw['model_size'], method=raw['method'],
                   seed=int(raw['seed']) if raw['seed'] else '', step=int(raw['checkpoint_step']))
        for g in GROUPS:
            row[g+'_accuracy'] = float(raw['v4_'+g])
            row[g+'_format'] = float(raw['v4_'+g+'_format'])
        if row['method'] == 'base':
            row.update(source='reference/per_model.csv', validation_format_step100='',
                       training_format_step100='', retained=True, exclusion_reason='')
        else:
            log = RESULTS / 'reference/training_logs' / row['model_size'].lower() / f"{row['method']}_s{row['seed']}.jsonl"
            finish_row(row, read_metrics(log), 'reference/per_model.csv')
        rows.append(row)
    return rows


def local_rows():
    rows = []
    for raw in read_csv(OUT / 'inputs/local_3b_seed0_evaluation.csv'):
        run_id = f"haotian_3b_{raw['method']}_s0_save10"
        row = dict(run_id=run_id, model_size='3B', method=raw['method'], seed=0,
                   step=int(raw['step']), **{m: float(raw[m]) for m in METRICS})
        rows.append(finish_row(row, read_metrics(OUT / 'inputs' / (run_id+'.metrics.jsonl')),
                              'inputs/local_3b_seed0_evaluation.csv'))
    return rows


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['model_size'], row['method'], row['step']].append(row)
    result = []
    for (size, method, step), values in groups.items():
        assert len({v['seed'] for v in values}) == len(values), (size, method, step)
        row = dict(model_size=size, method=method, step=step, n=len(values),
                   seeds=','.join(str(v['seed']) for v in sorted(values, key=lambda v: str(v['seed']))))
        for m in METRICS:
            scores = [v[m] for v in values]
            row[m+'_mean'] = st.mean(scores)
            row[m+'_sd'] = st.stdev(scores) if len(scores) > 1 else ''
        result.append(row)
    return result


def audit(rows, unique_key):
    keys = [tuple(r[k] for k in unique_key) for r in rows]
    assert len(keys) == len(set(keys)), unique_key
    max_error = 0.
    for r in rows:
        assert all(0 <= r[m] <= 100 for m in METRICS), r['run_id']
        for metric in ['accuracy', 'format']:
            expected = st.mean(r[g+'_'+metric] for g in GROUPS[:3])
            error = abs(expected-r['average_'+metric])
            assert error < 1e-8, (r['run_id'], metric)
            max_error = max(max_error, error)
    return max_error


def cell(row, metric, math=False, bold=False):
    mean, sd = row[metric+'_mean'], row[metric+'_sd']
    value = f'{mean:.2f}'
    if math:
        value = r'\mathbf{'+value+'}' if bold else value
        value += (r'_{\pm '+f'{sd:.2f}'+'}') if sd != '' else ''
        return '$'+value+'$'
    return value + (f' ± {sd:.2f}' if sd != '' else '')


def save_figure(fig, name):
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def draw_table(stats):
    fig, ax = plt.subplots(figsize=(15.6, 7.0))
    ax.set_xlim(0, 10.8)
    ax.set_ylim(-1.5, 14.6)
    ax.axis('off')
    xcols = [2.45+i*1.1 for i in range(8)]
    ax.hlines(14, -.05, 10.6, lw=1.5, color='black')
    for i, name in enumerate(['Live', 'Non-Live', 'Multi-Turn', 'Average']):
        center = st.mean(xcols[2*i:2*i+2])
        ax.text(center, 13.55, name, ha='center', va='center', fontsize=16, fontfamily='DejaVu Serif')
        ax.hlines(13.15, xcols[2*i]-.48, xcols[2*i+1]+.48, lw=.8, color='.3')
    ax.text(.15, 12.65, 'Size', va='center', fontsize=14, fontfamily='DejaVu Serif')
    ax.text(.63, 12.65, 'Method', va='center', fontsize=14, fontfamily='DejaVu Serif')
    for x, label in zip(xcols, ['Acc.', 'Format']*4):
        ax.text(x, 12.65, label, ha='center', va='center', fontsize=14, fontfamily='DejaVu Serif')
    ax.hlines(12.2, -.05, 10.6, lw=1, color='black')
    ordered = []
    for size in ['1.5B', '3B']:
        selected = {r['method']: r for r in stats if r['model_size']==size}
        best = {m: max(selected[k][m+'_mean'] for k in METHODS) for m in METRICS}
        for method in ['base']+METHODS:
            ordered.append((selected[method], best))
    for i, (row, best) in enumerate(ordered):
        y = 11.75-i*.91
        if i in [0, 6]:
            ax.text(.02, y, row['model_size'], va='center', fontsize=14, fontfamily='DejaVu Serif')
        ax.text(.63, y, LABELS[row['method']], va='center', fontsize=14, fontfamily='DejaVu Serif')
        for x, metric in zip(xcols, METRICS):
            b = row['method'] != 'base' and round(row[metric+'_mean'],2)==round(best[metric],2)
            ax.text(x, y, cell(row,metric,math=True,bold=b), ha='center', va='center', fontsize=13.7)
        if i==5:
            ax.hlines(y-.455, -.05, 10.6, lw=.9, color='black')
    ax.hlines(1.22, -.05, 10.6, lw=1.5, color='black')
    ax.text(5.25, .55, 'BFCL V4 · step 100 · mean ± sample SD (%)', ha='center', fontsize=12)
    ax.text(5.25, -.05, 'GRPO / GDPO / DARA: 5 seeds per size; DVAO / GD²PO-Hard: 3 retained seeds per size.', ha='center', fontsize=10.5)
    ax.text(5.25, -.57, 'Retention: step-100 RLLA validation Format ≥ 80%. Average = mean over Live, Non-Live, and Multi-Turn.', ha='center', fontsize=10.5)
    save_figure(fig, 'table1_bfcl_v4')
    caption = ('Performance on BFCL V4 at step 100. Acc. denotes tool-calling accuracy and Format denotes output-format compliance. '
               'Avg. is the mean over Live, Non-Live, and Multi-Turn. Percentages are reported as mean $\\pm$ sample standard deviation. '
               'We retain runs with step-100 RLLA validation Format $\\geq80\\%$. GRPO, GDPO, and DARA use five seeds per size; '
               'DVAO and GD$^2$PO-Hard use three retained seeds per size. Base uses one evaluation.')
    lines = [r'\begin{table*}[t]', r'\centering', r'\caption{'+caption+'}', r'\label{tab:bfcl-v4}',
             r'\setlength{\tabcolsep}{4pt}', r'\small', r'\begin{tabular}{llrrrrrrrr}', r'\toprule',
             r' & & \multicolumn{2}{c}{Live} & \multicolumn{2}{c}{Non-Live} & \multicolumn{2}{c}{Multi-Turn} & \multicolumn{2}{c}{Average} \\',
             r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}',
             r'Size & Method & Acc. & Format & Acc. & Format & Acc. & Format & Acc. & Format \\', r'\midrule']
    for i,(row,best) in enumerate(ordered):
        label = r'GD$^2$PO-Hard' if row['method']=='gd2po_hard' else PLAIN[row['method']]
        cells = [row['model_size'] if i in [0,6] else '', label]
        for m in METRICS:
            bold = row['method']!='base' and round(row[m+'_mean'],2)==round(best[m],2)
            cells.append(cell(row,m,math=True,bold=bold))
        lines.append(' & '.join(cells)+r' \\')
        if i==5: lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    (OUT/'table1_bfcl_v4.tex').write_text('\n'.join(lines)+'\n')


def draw_curves(stats, base):
    lookup = {(r['method'],r['step']):r for r in stats if r['model_size']=='1.5B'}
    steps = np.arange(0,101,10)
    def panel(ax, metric, title):
        for method in METHODS:
            records = [lookup[method,int(s)] for s in steps[1:]]
            ys = np.array([base[metric]]+[r[metric+'_mean'] for r in records])
            ax.plot(steps,ys,'o-',ms=4,lw=2.1,color=COLORS[method],label=LABELS[method],zorder=4 if method=='dara' else 3)
        ax.set_title(title,fontsize=13,fontweight='bold',pad=10)
        ax.set_xlabel('Training Steps',fontsize=12)
        ax.set_ylabel('Average Accuracy (%)' if metric=='average_accuracy' else 'Average Format (%)',fontsize=12)
        ax.set_xlim(0,100)
        ax.set_xticks(np.arange(0,101,20))
        ax.set_ylim((0,56) if metric=='average_accuracy' else (0,102))
        ax.grid(True,color='.82',alpha=.8)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=10)
    note = 'GRPO / DARA: n=3; GDPO: n=2; DVAO: seed 3 (n=1); Hard: seed 4 (n=1). Step 0: Base.'
    fig,axs = plt.subplots(1,2,figsize=(12.5,4.7))
    panel(axs[0],'average_accuracy','(a) BFCL V4 Average Accuracy')
    panel(axs[1],'average_format','(b) BFCL V4 Average Format')
    handles,labels=axs[0].get_legend_handles_labels()
    fig.suptitle('Qwen2.5-1.5B-Instruct | BFCL V4',fontsize=15,fontweight='bold',y=.99)
    fig.legend(handles,labels,ncol=5,loc='upper center',bbox_to_anchor=(.5,.92),frameon=False,fontsize=10)
    fig.text(.5,.005,note,ha='center',fontsize=9)
    fig.subplots_adjust(top=.75,bottom=.17,wspace=.23)
    save_figure(fig,'bfcl_v4_checkpoint_dynamics')
    for metric,title in [('average_accuracy','BFCL V4 Average Accuracy'),('average_format','BFCL V4 Average Format')]:
        fig,ax=plt.subplots(figsize=(7,4.8))
        panel(ax,metric,title)
        ax.legend(loc='lower right',frameon=True,fontsize=10)
        fig.text(.5,-.015,'Qwen2.5-1.5B-Instruct · mean across retained runs',ha='center',fontsize=10)
        fig.tight_layout()
        save_figure(fig,'bfcl_v4_'+metric)


def rlla_curve_rows(selected_runs):
    rows = []
    for run_id in sorted({r['run_id'] for r in selected_runs}):
        run = next(r for r in selected_runs if r['run_id'] == run_id)
        log = RESULTS / 'rental_1p5b_20260919/training/runs' / run_id / 'metrics.jsonl'
        for point in read_metrics(log):
            if point['step'] not in range(0, 101, 10):
                continue
            rows.append(dict(run_id=run_id, model_size='1.5B', method=run['method'], seed=int(run['seed']),
                             step=point['step'], correctness=point['val/test_correctness/rlla'],
                             format=100 * point['val/test_format/rlla']))
    return rows


def aggregate_rlla(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['method'], row['step']].append(row)
    result = []
    for (method, step), values in groups.items():
        row = dict(model_size='1.5B', method=method, step=step, n=len(values),
                   seeds=','.join(str(v['seed']) for v in sorted(values, key=lambda v: v['seed'])))
        for metric in ['correctness', 'format']:
            scores = [v[metric] for v in values]
            row[metric+'_mean'] = st.mean(scores)
            row[metric+'_sd'] = st.stdev(scores) if len(scores) > 1 else ''
        result.append(row)
    return result


def draw_rlla_curves(stats):
    lookup = {(r['method'], r['step']): r for r in stats}
    steps = np.arange(0, 101, 10)
    def panel(ax, metric, title):
        for method in METHODS:
            values = [lookup[method, int(step)][metric+'_mean'] for step in steps]
            ax.plot(steps, values, 'o-', ms=4, lw=2.1, color=COLORS[method], label=LABELS[method],
                    zorder=4 if method == 'dara' else 3)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Correctness Reward' if metric == 'correctness' else 'Format Compliance (%)', fontsize=12)
        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.set_ylim((-2.85, 2.1) if metric == 'correctness' else (0, 102))
        ax.grid(True, color='.82', alpha=.8)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=10)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    panel(axes[0], 'correctness', '(a) Validation Correctness Reward')
    panel(axes[1], 'format', '(b) Validation Format Compliance')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle('Qwen2.5-1.5B-Instruct | RLLA Validation', fontsize=15, fontweight='bold', y=.99)
    fig.legend(handles, labels, ncol=5, loc='upper center', bbox_to_anchor=(.5, .92), frameon=False, fontsize=10)
    fig.text(.5, .005, 'GRPO / DARA: n=3; GDPO: n=2; DVAO: seed 3 (n=1); Hard: seed 4 (n=1).', ha='center', fontsize=9)
    fig.subplots_adjust(top=.75, bottom=.17, wspace=.23)
    save_figure(fig, 'rlla_validation_dynamics')
    for metric, title in [('correctness', 'RLLA Validation Correctness Reward'), ('format', 'RLLA Validation Format Compliance')]:
        fig, ax = plt.subplots(figsize=(7, 4.8))
        panel(ax, metric, title)
        ax.legend(loc='lower right', frameon=True, fontsize=10)
        fig.text(.5, -.015, 'Qwen2.5-1.5B-Instruct · mean across selected runs', ha='center', fontsize=10)
        fig.tight_layout()
        save_figure(fig, 'rlla_validation_'+metric)


def markdown_table(stats):
    lines = ['| Size | Method | Live Acc. | Live Format | Non-Live Acc. | Non-Live Format | Multi-Turn Acc. | Multi-Turn Format | Average Acc. | Average Format |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for size in ['1.5B','3B']:
        for method in ['base']+METHODS:
            r=next(x for x in stats if x['model_size']==size and x['method']==method)
            lines.append('| '+' | '.join([size,PLAIN[method]]+[cell(r,m) for m in METRICS])+' |')
    return '\n'.join(lines)


def main():
    historical = historical_rows()
    one = rental_rows('rental_1p5b_20260919')
    three = rental_rows('rental_3b_20260920') + local_rows()
    process = one + three
    final = historical + [r for r in one if r['step']==100 and r['method'] in ['dvao','gd2po_hard'] and r['seed'] in [4,5]] + [r for r in three if r['step']==100]
    final_error = audit(final,['model_size','method','seed'])
    process_error = audit(process,['model_size','method','seed','step'])
    assert len(one)==150 and len(three)==60
    per_run=defaultdict(list)
    for r in process: per_run[r['run_id']].append(r)
    assert all(sorted(r['step'] for r in rows)==list(range(10,101,10)) for rows in per_run.values())
    kept_final=[r for r in final if r['retained']]
    kept_process=[r for r in process if r['retained']]
    final_stats=aggregate(kept_final)
    process_stats=aggregate(kept_process)
    for name,rows in [('final_per_run_all.csv',final),('final_per_run.csv',kept_final),
                      ('final_method_results.csv',final_stats),('final_method_results_all_runs.csv',aggregate(final)),
                      ('checkpoint_per_run_all.csv',process),('checkpoint_per_run.csv',kept_process),
                      ('checkpoint_method_results.csv',process_stats),
                      ('checkpoint_method_results_all_runs.csv',aggregate(process)),
                      ('process_final_per_run.csv',[r for r in kept_process if r['step']==100]),
                      ('process_final_method_results.csv',aggregate([r for r in kept_process if r['step']==100]))]:
        save_csv(name,rows)
    admissions=[]
    for cohort,rows in [('final',final),('checkpoint',process)]:
        for r in rows:
            if r['step']==100:
                admissions.append(dict(cohort=cohort,**{k:r[k] for k in ['run_id','model_size','method','seed','validation_format_step100','training_format_step100','retained','exclusion_reason','source']}))
    save_csv('run_selection.csv',admissions)
    save_csv('excluded_runs.csv',[r for r in admissions if not r['retained']])
    draw_table(final_stats)
    selected_curves = curve_rows(kept_process)
    curve_stats = aggregate(selected_curves)
    rlla_rows = rlla_curve_rows(selected_curves)
    rlla_stats = aggregate_rlla(rlla_rows)
    for name, rows in [('curve_per_run.csv', selected_curves), ('curve_method_results.csv', curve_stats),
                       ('rlla_per_run.csv', rlla_rows), ('rlla_method_results.csv', rlla_stats)]:
        save_csv(name, rows)
    draw_curves(curve_stats,next(r for r in historical if r['method']=='base' and r['model_size']=='1.5B'))
    draw_rlla_curves(rlla_stats)
    (OUT/'TABLE1.md').write_text(markdown_table(final_stats)+'\n')
    caption = ('Checkpoint-wise BFCL V4 evaluation of Qwen2.5-1.5B-Instruct. The panels show Average Accuracy and Average Format, '
               'each averaged over Live, Non-Live, and Multi-Turn. Both panels show mean curves. '
               'GRPO uses seeds 0/2/5, GDPO uses 0/1, and DARA uses 0/1/2. DVAO and GD²PO-Hard show selected seeds 3 and 4 respectively, '
               'chosen for earlier Format convergence among their completed process runs. Checkpoints are evaluated every 10 training steps; step 0 uses the shared base model evaluation.')
    (OUT/'FIGURE_CAPTION.txt').write_text(caption+'\n')
    report=dict(pr=1,pr_commit=PR_COMMIT,final_rows_all=len(final),final_rows_retained=len(kept_final),
                checkpoint_rows_all=len(process),checkpoint_rows_retained=len(kept_process),
                checkpoint_steps=list(range(10,101,10)),retention_metric='step100 val/test_format/rlla',
                retention_threshold=MIN_FORMAT,group_average_max_error=max(final_error,process_error),
                unique_final_size_method_seed=True,complete_process_trajectories=True)
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(markdown_table(final_stats))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
