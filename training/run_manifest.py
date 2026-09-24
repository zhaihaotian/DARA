"""Execute one named four-GPU run from the handoff experiment matrix."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--id', required=True)
    p.add_argument('--model-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    rows = json.loads(a.manifest.read_text())['runs']
    row = next(r for r in rows if r['id'] == a.id)
    if row['status'] == 'reuse_complete':
        raise ValueError('This G=4 reference is already complete; use the supplied reference results.')
    model = a.model_root/('Qwen2.5-1.5B-Instruct' if row['model_size']=='1.5b' else 'Qwen2.5-3B-Instruct')
    command = [sys.executable, str(Path(__file__).with_name('launch.py')), '--method', row['method'],
               '--seed', str(row['seed']), '--model-size', row['model_size'], '--model', str(model),
               '--group-size', str(row['group_size']), '--rewards', row['rewards'],
               '--output', str(a.output_root/row['id'])]
    if 'save_freq' in row:
        command += ['--save-freq', str(row['save_freq'])]
    if 'length_min_words' in row:
        command += ['--length-min-words', str(row['length_min_words'])]
    if a.dry_run:
        command.append('--dry-run')
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
