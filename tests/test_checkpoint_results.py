import json
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation'))
from checkpoint_results import collect


class CheckpointResultsTests(unittest.TestCase):
    def test_seed3_final_is_exported_and_steps_are_aggregated_separately(self):
        manifest = json.loads((Path(__file__).resolve().parents[1] / 'configs/rental_1p5b_20h.json').read_text())
        self.assertEqual(manifest['runs'][14]['id'], 'haotian_1p5b_dvao_s3_save10')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for seed, step, value in [(3, 10, .1), (3, 100, .6), (4, 100, .8)]:
                out = root / f'haotian_1p5b_dvao_s{seed}_save10' / f'step_{step}'
                out.mkdir(parents=True)
                summary = dict(version='v4', n_cases=3301,
                               groups={metric: dict.fromkeys(('live', 'non_live', 'multi_turn', 'average'), value)
                                       for metric in ('accuracy', 'format')})
                (out / 'summary.json').write_text(json.dumps(summary))
            result = collect(manifest, root)
            self.assertEqual(len(result['checkpoint_per_model']), 3)
            self.assertEqual(sorted(row['seed'] for row in result['process_final_per_model']), [3, 4])
            final = result['process_final_method_results'][0]
            self.assertEqual(final['expected_seeds'], 3)
            self.assertEqual(final['completed_seeds'], 2)
            self.assertEqual(final['average_accuracy_mean'], 70.)
            self.assertAlmostEqual(final['average_format_sd'], 200 ** .5)
            repo = Path(__file__).resolve().parents[1]
            command = [sys.executable, str(repo/'evaluation/checkpoint_results.py'),
                       '--manifest', str(repo/'configs/rental_1p5b_20h.json'),
                       '--evaluation-root', str(root), '--output', str(root/'report')]
            completed = subprocess.run(command, capture_output=True, text=True, check=True)
            progress = json.loads(completed.stdout)
            self.assertEqual(progress, dict(scored_checkpoints=3, expected_checkpoints=150,
                                            scored_finals=2, expected_finals=15))
            with (root/'report/process_final_per_model.csv').open() as stream:
                exported = list(csv.DictReader(stream))
            self.assertEqual(sorted(row['seed'] for row in exported), ['3', '4'])


if __name__ == '__main__':
    unittest.main()
