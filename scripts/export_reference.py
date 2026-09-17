"""Export Haotian reference tables and training dynamics from saved artifacts."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics as st

METHODS = ['base', 'grpo', 'gdpo', 'dara', 'rvpo', 'dvao', 'gd2po_hard', 'gdpo_saw']
LABELS = dict(zip(METHODS, ['Base', 'GRPO', 'GDPO', 'DARA', 'RVPO', 'DVAO', 'GD²PO-Hard', 'GDPO-SAW']))
METRICS = ['v3_live', 'v3_non_live', 'v4_live', 'v4_non_live', 'v4_multi_turn', 'v4_average', 'v4_average_format']


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--combined', type=Path, required=True)
    p.add_argument('--training-1p5b', type=Path, required=True)
    p.add_argument('--training-3b', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'results/reference')
    a = p.parse_args(); out = a.output; out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(a.combined.open()) if r['infra']=='haotian']
    for r in rows:
        r['source_model_id'] = r.pop('model')
        r['method'] = 'dara' if r['method']=='rdgdpo_symmetric' else r['method']
        r['run_id'] = f"{r['model_size'].lower()}-{r['method']}-s{r['seed']}"
    rows = [r for r in rows if r['method'] in METHODS]
    rows.sort(key=lambda r:(r['model_size'], METHODS.index(r['method']),r['seed']))
    csv_write(out/'per_model.csv', rows)
    for selected in [False, True]:
        groups = defaultdict(list)
        for r in rows:
            if selected and (r['model_size'],r['method'],r['seed']) in [('1.5B','dvao','1'),('1.5B','gd2po_hard','2')]:
                continue
            groups[(r['model_size'],r['method'])].append(r)
        aggregates = []
        md = ['# '+('Selected-seed diagnostic' if selected else 'All completed seeds'), '',
              '单位：百分数，mean ± sample standard deviation (ddof=1)。Base 为一次固定解码评测。', '',
              ('此附表事后排除 DVAO seed 1、GD²PO-Hard seed 2 的 format 未收敛 run；不能称为三 seed 结果。完整结果见 ALL_SEEDS.md。'
               if selected else '包含每个已完成 seed；DVAO seed 1、GD²PO-Hard seed 2 的 format 未收敛结果也保留。'), '',
              '| Model | Method | n | V3 Live AST | V3 Non-Live AST | V4 Live AST | V4 Non-Live AST | V4 Multi-Turn | V4 Avg | V4 Avg Format* |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for (size,method), runs in groups.items():
            row = dict(model_size=size,method=method,n=len(runs),seeds=','.join(r['seed'] for r in runs))
            cells=[]
            for key in METRICS:
                vals = [float(r[key]) for r in runs]
                mean=st.mean(vals); sd=st.stdev(vals) if len(vals)>1 else 0
                row[key+'_mean']=mean; row[key+'_std']=sd
                cells.append(f'{mean:.2f} ± {sd:.2f}' if len(vals)>1 else f'{mean:.2f}')
            aggregates.append(row)
            md.append('| '+' | '.join([size,LABELS[method],str(len(runs)),*cells])+' |')
        suffix='SELECTED_SEEDS' if selected else 'ALL_SEEDS'
        csv_write(out/(suffix.lower()+'.csv'),aggregates)
        (out/(suffix+'.md')).write_text('\n'.join(md)+'\n')
    dynamics=[]; status=[]; provenance=[]
    for row in rows:
        if row['method']=='base': continue
        method=row['method'];seed=row['seed'];size=row['model_size']
        old = ('rdgdpo' if size=='1.5B' else 'rdgdpo_symmetric') if method=='dara' else method
        directory=(a.training_1p5b if size=='1.5B' else a.training_3b)/f'{old}_seed{seed}'
        source=directory/'metrics.jsonl'
        records=[json.loads(l) for l in source.read_text().splitlines()]
        steps=[r['step'] for r in records if 'actor/pg_loss' in r]
        if steps!=list(range(1,101)): raise ValueError(f'Incomplete training: {source}')
        normalized=[{k.replace('rdgdpo/','dara/'):v for k,v in r.items()} for r in records]
        log_dir=out/'training_logs'/size.lower();log_dir.mkdir(parents=True,exist_ok=True)
        target=log_dir/f'{method}_s{seed}.jsonl'
        target.write_text(''.join(json.dumps(r)+'\n' for r in normalized))
        for r in normalized:
            d=dict(model_size=size,method=method,seed=seed,step=r['step'])
            mapping={'critic/format_score/mean':'train_format','critic/correctness_score/mean':'train_correctness',
                     'critic/length_score/mean':'train_length','val/test_format/rlla':'val_format',
                     'val/test_correctness/rlla':'val_correctness','response_length/mean':'response_tokens',
                     'timing_s/step':'step_seconds','actor/grad_norm':'grad_norm'}
            for key,name in mapping.items():
                if key in r:d[name]=r[key]
            d.update({k:v for k,v in r.items() if k.startswith('dara/')})
            dynamics.append(d)
        hits=[r['step'] for r in records if r.get('critic/format_score/mean',0)>=.8]
        final=records[-1]
        status.append(dict(model_size=size,method=method,seed=seed,step=100,
                           format_t80=hits[0] if hits else '', final_train_format=final['critic/format_score/mean'],
                           final_val_format=final.get('val/test_format/rlla',''),
                           total_training_seconds=sum(r.get('timing_s/step',0) for r in records)))
        provenance.append(dict(run_id=row['run_id'],source_metrics=str(source),
                               checkpoint=str(directory/'actor/global_step_100'),export=str(target.relative_to(out))))
    csv_write(out/'training_dynamics.csv',dynamics)
    csv_write(out/'training_status.csv',status)
    (out/'local_artifact_locations.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(f'Exported {len(rows)} evaluated models and {len(status)} complete training runs.')


if __name__=='__main__': main()
