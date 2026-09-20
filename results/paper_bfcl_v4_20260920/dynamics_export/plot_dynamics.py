"""Draw BFCL V4 checkpoint curves from the accompanying plot_data.csv."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
METHODS = ['grpo', 'gdpo', 'dvao', 'gd2po_hard', 'dara']
LABELS = {'base': 'Base', 'grpo': 'GRPO', 'gdpo': 'GDPO', 'dvao': 'DVAO', 'gd2po_hard': 'GD$^2$PO-Hard', 'dara': 'DARA'}
COLORS = {'grpo': '#888888', 'gdpo': '#278e2b', 'dvao': '#f18420', 'gd2po_hard': '#9665c9', 'dara': '#2563df'}


def save_figure(fig, name):
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)


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
