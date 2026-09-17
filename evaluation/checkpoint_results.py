"""Export BFCL V4 checkpoint curves and their step100 final results."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

METRICS = [f'{group}_{metric}' for group in ('live', 'non_live', 'multi_turn', 'average')
           for metric in ('accuracy', 'format')]
ROW_FIELDS = ['run_id', 'model_size', 'method', 'seed', 'step', 'summary'] + METRICS
STAT_FIELDS = ['model_size', 'method', 'step', 'completed_seeds', 'expected_seeds', 'seeds'] + [
    f'{metric}_{stat}' for metric in METRICS for stat in ('mean', 'sd')]


def collect(manifest, evaluation_root):
    evaluation = manifest['evaluation']
    rows, expected = [], defaultdict(int)
    for run in manifest['runs']:
        for step in manifest['checkpoint_steps']:
            expected[run['model_size'], run['method'], step] += 1
            path = evaluation_root / run['id'] / f'step_{step}' / 'summary.json'
            if not path.exists():
                continue
            summary = json.loads(path.read_text())
            if summary['version'] != evaluation['version'] or summary['n_cases'] != 3301:
                raise ValueError(f'Expected complete BFCL V4 results: {path}')
            values = {f'{group}_{metric}': score * 100
                      for metric in ('accuracy', 'format')
                      for group, score in summary['groups'][metric].items()}
            rows.append(dict(run_id=run['id'], model_size=run['model_size'], method=run['method'],
                             seed=run['seed'], step=step, summary=str(path), **values))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['model_size'], row['method'], row['step']].append(row)
    stats = []
    for (size, method, step), values in grouped.items():
        result = dict(model_size=size, method=method, step=step, completed_seeds=len(values),
                      expected_seeds=expected[size, method, step],
                      seeds=','.join(str(row['seed']) for row in values))
        for metric in METRICS:
            scores = [row[metric] for row in values]
            result[metric + '_mean'] = statistics.mean(scores)
            result[metric + '_sd'] = statistics.stdev(scores) if len(scores) > 1 else None
        stats.append(result)
    final_step = evaluation['final_step']
    return dict(checkpoint_per_model=rows, checkpoint_method_results=stats,
                process_final_per_model=[row for row in rows if row['step'] == final_step],
                process_final_method_results=[row for row in stats if row['step'] == final_step])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--evaluation-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    records = collect(manifest, args.evaluation_root)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in records.items():
        with (args.output / f'{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=STAT_FIELDS if 'method_results' in name else ROW_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(dict(scored_checkpoints=len(records['checkpoint_per_model']),
                          expected_checkpoints=len(manifest['runs']) * len(manifest['checkpoint_steps']),
                          scored_finals=len(records['process_final_per_model']),
                          expected_finals=len(manifest['runs']))))


if __name__ == '__main__':
    main()
