"""Export both decoding temperatures of the length>=64 experiment."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

GROUPS = ('live', 'non_live', 'multi_turn', 'average')
METRICS = ('accuracy', 'format', 'length', 'length_ge_64', 'think_words')


def collect(manifest, evaluation_root):
    rows, expected = [], defaultdict(int)
    for run in manifest['runs']:
        for profile in manifest['evaluation']['profiles']:
            key = (run['method'], profile['id'])
            expected[key] += 1
            directory = evaluation_root / profile['id'] / run['id']
            path = directory / 'summary.json'
            if not path.exists():
                continue
            summary = json.loads(path.read_text())
            inference = json.loads((directory / 'inference.json').read_text())
            if (summary['version'], summary['n_cases'], summary.get('length_min_words')) != ('v4', 3301, 64):
                raise ValueError(f'Expected complete V4 length>=64 results: {path}')
            if (inference['temperature'], inference['top_p'], inference.get('length_min_words')) != (
                    profile['temperature'], profile['top_p'], 64):
                raise ValueError(f'Inference settings do not match the profile: {directory}')
            row = dict(run_id=run['id'], model_size=run['model_size'], method=run['method'],
                       seed=run['seed'], step=100, profile=profile['id'],
                       temperature=profile['temperature'], top_p=profile['top_p'], summary=str(path))
            for group in GROUPS:
                for metric in METRICS:
                    value = summary['groups'][metric][group]
                    row[f'{group}_{metric}'] = value * 100 if metric in ('accuracy', 'format', 'length_ge_64') else value
            row['three_reward_overall'] = (row['average_accuracy'] + row['average_format']) / 100 + row['average_length']
            rows.append(row)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['method'], row['profile']].append(row)
    stats = []
    metrics = [f'{group}_{metric}' for group in GROUPS for metric in METRICS] + ['three_reward_overall']
    for (method, profile), samples in grouped.items():
        row = dict(method=method, profile=profile, model_size=samples[0]['model_size'],
                   temperature=samples[0]['temperature'], top_p=samples[0]['top_p'],
                   completed_seeds=len(samples), expected_seeds=expected[method, profile],
                   seeds=','.join(str(sample['seed']) for sample in samples))
        for metric in metrics:
            values = [sample[metric] for sample in samples]
            row[metric + '_mean'] = statistics.mean(values)
            row[metric + '_sd'] = statistics.stdev(values) if len(values) > 1 else None
        stats.append(row)
    return rows, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/length_ge64_binary.json')
    parser.add_argument('--evaluation-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    models, methods = collect(manifest, args.evaluation_root)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in [('per_model', models), ('method_results', methods)]:
        if rows:
            with (args.output / f'{name}.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    print(json.dumps(dict(completed_evaluations=len(models),
                          expected_evaluations=len(manifest['runs']) * len(manifest['evaluation']['profiles']))))


if __name__ == '__main__':
    main()
