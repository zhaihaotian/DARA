"""Explain the saved final, trajectory, and step-60 seed selections."""
import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
RESULTS = OUT.parent
METHODS = ['grpo', 'gdpo', 'dvao', 'gd2po_hard', 'dara']
LABELS = dict(grpo='GRPO', gdpo='GDPO', dvao='DVAO', gd2po_hard='GD²PO-Hard', dara='DARA')


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def save(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percent(value):
    return '—' if value == '' else f'{float(value)*100:.2f}%'


def markdown(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows])


def main():
    final = [r for r in read(OUT/'final_per_run_all.csv') if r['method'] != 'base']
    process = read(OUT/'checkpoint_per_run_all.csv')
    step60 = read(OUT/'step60/selection.csv')
    final_map = {r['run_id']: r for r in final}
    process_map = {r['run_id']: r for r in process if int(r['step']) == 100}
    curve_ids = {r['run_id'] for r in read(OUT/'curve_per_run.csv')}
    step60_map = {r['run_id']: r for r in step60}
    all_runs = {**final_map, **process_map}
    local_paths = {r['run_id']:r for r in json.loads((RESULTS/'reference/local_artifact_locations.json').read_text())}
    ledger, checkpoint_status = [], []
    for run_id, r in all_runs.items():
        src = r['source']
        if src == 'reference/per_model.csv':
            log = RESULTS/'reference/training_logs'/r['model_size'].lower()/(r['method']+'_s'+r['seed']+'.jsonl')
            provenance = '本机原始训练'
            original_log = local_paths[run_id]['source_metrics']
        elif src.startswith('inputs/'):
            log = OUT/'inputs'/(run_id+'.metrics.jsonl')
            provenance = '本机3B过程训练'
            original_log = str(log)
        else:
            log = RESULTS/src.split('/')[0]/'training/runs'/run_id/'metrics.jsonl'
            provenance = 'PR租机1.5B' if r['model_size']=='1.5B' else 'PR租机3B'
            original_log = str(log)
        records = sorted([json.loads(line) for line in log.read_text().splitlines() if line.strip()], key=lambda x:x['step'])
        by_step = {x['step']:x for x in records}
        def status(mapping):
            return ('纳入' if mapping[run_id]['retained']=='True' else '未收敛，排除') if run_id in mapping else '未选入这组实验'
        row = dict(run_id=run_id, model_size=r['model_size'], method=r['method'], seed=int(r['seed']),
                   provenance=provenance, final_table=status(final_map), process_curve=status(process_map), step60_table=status(step60_map),
                   validation_format_step60=by_step.get(60,{}).get('val/test_format/rlla',''),
                   validation_format_step100=by_step[100]['val/test_format/rlla'],
                   training_format_step60=by_step.get(60,{}).get('critic/format_score/mean',''),
                   training_format_step100=by_step[100]['critic/format_score/mean'],
                   first_validation_format_ge_0p8=next((x['step'] for x in records if x.get('val/test_format/rlla',0)>=.8),''),
                   first_training_format_ge_0p8=next((x['step'] for x in records if x.get('critic/format_score/mean',0)>=.8),''),
                   bfcl_v4_final_average_accuracy=r['average_accuracy'], bfcl_v4_final_average_format=r['average_format'],
                   bfcl_v4_step60_average_accuracy=step60_map.get(run_id,{}).get('average_accuracy',''),
                   bfcl_v4_step60_average_format=step60_map.get(run_id,{}).get('average_format',''),
                   evaluation_source=src, metrics_snapshot=str(log.relative_to(RESULTS)), original_metrics=original_log)
        if row['model_size']=='1.5B' and row['process_curve']=='纳入' and run_id not in curve_ids:
            row['process_curve']='未选入过程图'
        ledger.append(row)
        for p in [x for x in process if x['run_id']==run_id]:
            m=by_step[int(p['step'])]
            checkpoint_status.append(dict(run_id=run_id, model_size=r['model_size'], method=r['method'], seed=int(r['seed']),
                step=int(p['step']), validation_format=m['val/test_format/rlla'], training_format=m['critic/format_score/mean'],
                converged_at_this_step=m['val/test_format/rlla']>=.8,
                in_existing_process_curve=row['process_curve']=='纳入',
                bfcl_v4_average_accuracy=p['average_accuracy'], bfcl_v4_average_format=p['average_format'],
                bfcl_v4_live_format=p['live_format'], bfcl_v4_non_live_format=p['non_live_format'], bfcl_v4_multi_turn_format=p['multi_turn_format']))

    # Match the ledger to the actual saved artifacts so no duplicate run or missing seed is reported.
    assert len(ledger)==57 and len({r['run_id'] for r in ledger})==57
    assert sum(r['final_table']=='纳入' for r in ledger)==42
    assert sum(r['process_curve']=='纳入' for r in ledger)==16
    assert sum(r['step60_table']=='纳入' for r in ledger)==16
    assert len(checkpoint_status)==210
    assert all((r['validation_format_step100']>=.8)==(r['final_table']=='纳入') for r in ledger if r['run_id'] in final_map)
    assert all((r['validation_format_step60']>=.8)==(r['step60_table']=='纳入') for r in ledger if r['run_id'] in step60_map)
    ledger.sort(key=lambda r:(r['model_size'], METHODS.index(r['method']), r['seed'], r['provenance']))
    save('seed_provenance.csv', ledger)
    save('checkpoint_seed_status.csv', sorted(checkpoint_status, key=lambda r:(r['model_size'], METHODS.index(r['method']), r['seed'], r['step'])))

    doc=['# 当前 BFCL V4 结果：来源、seed 与收敛状态', '',
         '本报告描述当前已经生成的 final 表、1.5B 过程图与 step60 汇总。step60图表展示1.5B，CSV保留两个规模的现有数据；下表的step60列表示数据纳入状态。全部使用 Haotian infra；PR 数据是合作者在租机上重新训练、评测后提交的结果。PR 来源对应 #1 的 a954110 提交。', '',
         '**final 表使用 step100 模型；过程图使用带完整过程 checkpoint 的独立 run；step60 汇总使用这些过程 run 的 step60 模型。相同 seed 编号必须连同 run ID 和来源一起看。**', '',
         '目前的收敛筛选指标为 RLLA validation Format，阈值0.8。final 表看 step100；1.5B过程图固定使用GRPO 0/2/5、GDPO 0/1、DARA 0/1/2、DVAO 3和Hard 4，绘制各run完整轨迹。DVAO与Hard各选取一个更早Format收敛的seed，n=1，RLLA与BFCL统一。step60 表按 step60 单独筛选。', '',
         '图里的 Average Format 是 BFCL V4 的 Live、Non-Live、Multi-Turn Format 算术平均；RLLA validation Format 是训练日志中的筛选指标。', '',
         '## Final 表：实际纳入的结果', '']
    summary=[]
    for size in ['1.5B','3B']:
        for method in METHODS:
            group=[r for r in ledger if r['model_size']==size and r['method']==method and r['run_id'] in final_map]
            kept=[r for r in group if r['final_table']=='纳入']
            origins=[]
            for origin in sorted({r['provenance'] for r in kept}):
                origins.append(origin+'：'+','.join(str(r['seed']) for r in kept if r['provenance']==origin))
            removed=[str(r['seed'])+' ('+percent(r['validation_format_step100'])+')' for r in group if r['final_table']!='纳入']
            summary.append([size,LABELS[method],'；'.join(origins),len(kept),'；'.join(removed) or '无'])
    doc.append(markdown(['规模','方法','纳入seed及来源','n','未收敛排除：seed（step100 validation Format）'],summary))
    doc += ['', 'Base 为本机对应规模的未训练模型，各一份固定评测。final 表纳入42个训练run和两个Base。', '',
            'GRPO、GDPO、DARA的final来自本机原始五seed；PR过程重跑的step100结果保存在process_final_per_run.csv，与各自过程轨迹对应。DVAO的历史final使用0/1/2及补齐4/5这五个候选seed；PR seed3的step100已评测并保存在过程final文件中，seed3承担过程比较。', '',
            '本机原始1.5B DVAO seed2：step100 validation Format为96.25%，训练batch Format为39.26%，整个训练尚未出现batch Format≥80%。当前final按照validation阈值将它纳入；它是到最后一次验证才达标的run。', '',
            '## 交给租机agent的过程实验', '',
            '1.5B共15次训练、150份checkpoint评测，全部完成；3B共4次训练、40份checkpoint评测，全部完成。3B再合入本机DVAO/Hard各一个seed0，共60份过程评测。完成100steps与达到Format阈值分别记录。', '',
            '1.5B原始GRPO/GDPO/DARA各五seed按训练Format首次达到0.8的step排序，按原交接约定去掉最快和最慢，重跑中间三个：GRPO 0/2/5，GDPO 0/1/5，DARA 0/1/2。DVAO过程使用3/4/5，Hard使用0/4/5。选择记录见../../docs/RENTAL_1P5B_20H.md。', '']
    process_rows=[]
    for r in ledger:
        if r['run_id'] not in process_map: continue
        process_rows.append([r['model_size'],LABELS[r['method']],r['seed'],r['provenance'],
                             r['first_validation_format_ge_0p8'] or '未达标',percent(r['validation_format_step60']),percent(r['validation_format_step100']),
                             r['process_curve'],r['step60_table']])
    doc.append(markdown(['规模','方法','seed','来源','首次validation≥80%的step','step60 validation Format','step100 validation Format','现有过程图/过程汇总','step60汇总'],process_rows))
    doc += ['', '3B的GRPO、GDPO、DARA各使用本机原始0/1/2/4/5，全部纳入final；当前只有step100模型。3B DVAO与Hard均使用本机seed0加PR seeds1/2，六个run都达标，且有完整10/20/…/100评测。', '',
            '## 过程图的Format与选取', '',
            '现有过程图的GRPO/DARA各n=3，GDPO为n=2，DVAO seed3和Hard seed4各n=1。每个选中run的完整轨迹都用于RLLA validation和BFCL V4过程图，样本数沿step保持不变。原始数据与final多seed汇总保留全部既定候选run及其筛选记录。', '',
            'DVAO seed3在step40首次validation达标，Hard seed4在step20达标，分别早于同方法另一个最终达标的过程run：DVAO seed4在step100、Hard seed0在step70。当前两套过程图固定选取DVAO seed3和Hard seed4。step60的BFCL Average Format分别为85.63%和90.09%。', '',
            '另外，已收敛模型在BFCL多轮Format上仍有差距：step60的GRPO Live/Non-Live约99.8%，Multi-Turn为24.48%，Average为74.71%；GDPO Multi-Turn为64.28%，Average为88.08%；DARA Multi-Turn为88.19%，Average为96.06%。因此BFCL Average Format的差距同时反映当前checkpoint的收敛状态和多轮格式表现。', '',
            '图的step100终点使用绘图seed集合。过程run的最终汇总process_final_method_results.csv继续使用DVAO 3/4、Hard 0/4，各n=2；历史final_method_results.csv使用本机及PR的既定多seed集合，DVAO为本机0/2加PR 4，Hard为本机0/1加PR 4，各n=3。', '',
            '## 本机DVAO seed0的额外复跑', '',
            '本机另有1.5B DVAO seed0 save10复跑，目录/scratch.global/lian0190/DARA/20260917/1p5b-dvao-g4-s0-two-save10。已完成100steps和十个checkpoint评测；step60及step100 validation Format均为8.75%，step100训练Format为7.13%。它未纳入当前final、过程图和step60表。原始本机DVAO seed0已达标并用于历史final；租机过程实验使用seed3。详情见../../docs/DVAO_SEED0_REPLAY.md。', '',
            '## 逐run完整台账', '',
            'seed_provenance.csv覆盖当前两批结果中的57个独立训练run，记录来源、三份图表是否纳入、step60/100 validation与training Format、首次达标step、BFCL最终值及原始日志位置。上述额外本机DVAO复跑单独记录在前一段。checkpoint_seed_status.csv覆盖210个过程评测点，分别记录当前step是否达标与是否进入现有过程图。', '']
    all_rows=[]
    for r in ledger:
        all_rows.append([r['run_id'],r['provenance'],percent(r['validation_format_step60']),percent(r['validation_format_step100']),
                         r['final_table'],r['process_curve'],r['step60_table']])
    doc.append(markdown(['Run ID','来源','step60 validation Format','step100 validation Format','final表','过程图/过程汇总','step60汇总'],all_rows))
    (OUT/'SEED_PROVENANCE.md').write_text('\n'.join(doc)+'\n')
    print('Wrote SEED_PROVENANCE.md, seed_provenance.csv (57 runs), checkpoint_seed_status.csv (210 checkpoints).')


if __name__=='__main__':
    main()
