"""Fetch the original ToolRL parquet split used by the Haotian experiments."""
import argparse
from pathlib import Path
import urllib.request

REVISION = '8cee13ec0ca72f0461da372a93a6fd8140dbb840'
SOURCE = f'https://raw.githubusercontent.com/qiancheng0/ToolRL/{REVISION}/dataset/rlla_4k'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[1]/'data/rlla_4k')
    args = p.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'test'):
        path = args.data_dir/f'{split}.parquet'
        with urllib.request.urlopen(f'{SOURCE}/{split}.parquet', timeout=120) as response:
            payload = response.read()
        temporary = path.with_suffix('.download')
        temporary.write_bytes(payload)
        temporary.replace(path)
        print(f'{split}: {len(payload)} bytes -> {path}')


if __name__ == '__main__':
    main()
