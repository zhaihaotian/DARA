"""Rejudge saved BFCL outputs and aggregate with the fixed protocol; no GPU."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import sys

from protocol import categories, expected_counts, aggregate, V3_PACKAGE, V4_COMMIT
from diagnostics import score_case


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version', choices=['v3', 'v4'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--length-min-words', type=int, default=None,
                   help='Length threshold; defaults to the saved inference.json setting')
    args = p.parse_args()
    out = args.output.resolve()
    inference_path = out/'inference.json'
    inference = json.loads(inference_path.read_text()) if inference_path.exists() else {}
    length_min_words = args.length_min_words if args.length_min_words is not None else inference.get('length_min_words', 0)
    if length_min_words < 0:
        p.error('--length-min-words must be nonnegative')
    metric_names = ['format', 'length', 'length_ge_512', 'think_words']
    if length_min_words and length_min_words != 512:
        metric_names.append(f'length_ge_{length_min_words}')
    os.environ['BFCL_PROJECT_ROOT'] = str(out)
    for key in ('MAX1STEP30MAX3', 'SCHEDULEREWARD', 'SCHEDULELENGTH'):
        os.environ[key] = '0'
    sys.path.insert(0, str(Path(__file__).parent / args.version))
    from adapter import register, RLLAHandler
    register()
    from bfcl_eval.eval_checker import eval_runner
    from bfcl_eval.utils import sort_key
    counts = expected_counts(args.version)
    by_category = {}
    for cat in categories(args.version):
        rows = [json.loads(line) for line in (out/'raw'/f'{cat}.jsonl').read_text().splitlines()]
        if len(rows) != counts[cat] or len({r['id'] for r in rows}) != counts[cat]:
            raise ValueError(f'{cat}: expected {counts[cat]} unique cases; inference is incomplete')
        by_category[cat] = sorted(rows, key=sort_key)
    if args.version == 'v4':
        from bfcl_eval.utils import get_directory_structure_by_category
        handler = RLLAHandler('rlla-eval', 0.6)
        for rows in by_category.values():
            handler.write(rows, out/'result', update_mode=True)
    else:
        result_dir = out/'result/rlla-eval'
        result_dir.mkdir(parents=True, exist_ok=True)
        for cat, rows in by_category.items():
            (result_dir/f'BFCL_v3_{cat}_result.json').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    eval_runner.main(['rlla-eval'], list(by_category), out/'result', out/'score')
    scores, diagnostics, per_case = {}, {}, []
    for cat, rows in by_category.items():
        directory = get_directory_structure_by_category(cat) if args.version == 'v4' else ''
        path = out/'score/rlla-eval'/directory/f'BFCL_{args.version}_{cat}_score.json'
        score = json.loads(path.read_text().splitlines()[0])
        if score['total_count'] != counts[cat]:
            raise ValueError(f'Official checker count mismatch: {cat}')
        scores[cat] = score
        ds = [dict(category=cat, **score_case(row, length_min_words=length_min_words)) for row in rows]
        per_case.extend(ds)
        diagnostics[cat] = {key: statistics.mean(d[key] for d in ds)
                            for key in metric_names}
    grouped = {'accuracy': aggregate({k:v['accuracy'] for k,v in scores.items()}, counts, args.version)}
    for metric in metric_names:
        grouped[metric] = aggregate({k:v[metric] for k,v in diagnostics.items()}, counts, args.version)
    with (out/'diagnostics_per_case.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_case[0]))
        writer.writeheader(); writer.writerows(per_case)
    result = dict(version=args.version, source=V3_PACKAGE if args.version=='v3' else V4_COMMIT,
                  n_cases=sum(counts.values()), categories=scores, diagnostics=diagnostics, groups=grouped)
    if length_min_words:
        result['length_min_words'] = length_min_words
    (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(grouped, indent=2))


if __name__ == '__main__':
    main()
