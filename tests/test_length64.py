import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'evaluation'))
from diagnostics import score_case
from length64_results import collect
from verl.utils.reward_score import rlla


def response(words):
    return '<think>' + 'word ' * words + '</think>\n<tool_call>\n[]\n</tool_call>'


class Length64Test(unittest.TestCase):
    def test_boundary_and_missing_tags_match_training_and_evaluation(self):
        texts = [response(n) for n in (0, 63, 64, 65, 512)]
        texts += ['<think>' + 'word ' * 64, 'word ' * 64 + '</think>']
        with patch.dict(os.environ, {'LENGTH_MIN_WORDS': '64', 'SCHEDULELENGTH': '0'}), contextlib.redirect_stdout(io.StringIO()):
            scores = rlla.customize_length_reward_func(
                [[{'content': text}] for text in texts], [''] * len(texts), 0, 1, 0)
            self.assertEqual(scores, [0, 0, 1, 1, 1, 0, 0])
            for text, expected in zip(texts, scores):
                evaluation = score_case(dict(id='simple_0', result=text), length_min_words=64)
                self.assertEqual(evaluation['length'], expected)
                self.assertEqual(evaluation['length_ge_64'], expected)
            # The recorded evaluation setting controls scoring independently of the shell.
            self.assertEqual(score_case(dict(id='simple_0', result=response(64)))['length'], .12)

    def test_full_reward_changes_only_the_length_channel(self):
        with patch.dict(os.environ, {'EXPERIMENT_NAME': 'qwen-test', 'WITHLENGTH': '1',
                                     'SCHEDULELENGTH': '0'}), contextlib.redirect_stdout(io.StringIO()):
            for words in (63, 64):
                text = response(words)
                with patch.dict(os.environ, {'LENGTH_MIN_WORDS': '0'}):
                    dense = rlla.compute_score(text, text)
                with patch.dict(os.environ, {'LENGTH_MIN_WORDS': '64'}):
                    binary = rlla.compute_score(text, text)
                self.assertEqual(dense[1:3], binary[1:3])
                self.assertEqual(binary[3], float(words >= 64))
                self.assertAlmostEqual(binary[0] - dense[0], binary[3] - dense[3])

    def test_multiturn_averages_emissions_before_cases(self):
        result = score_case(dict(id='multi_turn_base_0',
                                 result=[[response(63), response(64)], [response(65)]]), length_min_words=64)
        self.assertEqual(result['length'], 2/3)
        self.assertEqual(result['length_ge_64'], 2/3)
        self.assertEqual(result['think_words'], 64)

    def test_eight_run_manifest_reaches_launcher(self):
        path = REPO / 'configs/length_ge64_binary.json'
        manifest = json.loads(path.read_text())
        runs = manifest['runs']
        self.assertEqual(len(runs), 8)
        self.assertEqual({(r['method'], r['seed']) for r in runs},
                         {(method, seed) for method in ('grpo', 'gdpo', 'dara', 'rdgdpo_positive') for seed in (1, 2)})
        self.assertEqual([(r['temperature'], r['top_p']) for r in manifest['evaluation']['profiles']], [(0.6, 1.0), (1.0, 1.0)])
        for run in runs:
            command = [sys.executable, str(REPO/'training/run_manifest.py'), '--manifest', str(path),
                       '--id', run['id'], '--model-root', '/models', '--output-root', '/tmp/length64', '--dry-run']
            record = json.loads(subprocess.check_output(command, text=True))
            self.assertEqual(record['environment']['WITHLENGTH'], '1')
            self.assertEqual(record['environment']['LENGTH_MIN_WORDS'], '64')
            self.assertEqual(record['environment']['SCHEDULELENGTH'], '0')
            self.assertEqual(record['settings']['+algorithm.reward_channels'], ['correctness', 'format', 'length'])
            self.assertEqual(record['gpus'], 4)
            self.assertEqual(record['responses_per_step'], 2048)
            self.assertEqual(record['settings']['trainer.total_training_steps'], 100)
            self.assertEqual(record['settings']['trainer.save_freq'], 100)
            if run['method'] == 'rdgdpo_positive':
                self.assertEqual(record['settings']['+algorithm.dara.calibration'], 'positive')

    def test_report_keeps_temperatures_and_seeds_separate(self):
        manifest = json.loads((REPO/'configs/length_ge64_binary.json').read_text())
        manifest['runs'] = [r for r in manifest['runs'] if r['method'] == 'gdpo']
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for run in manifest['runs']:
                for profile in manifest['evaluation']['profiles']:
                    out = root / profile['id'] / run['id']
                    out.mkdir(parents=True)
                    value = .2 * run['seed'] if profile['temperature'] == .6 else .1 * run['seed']
                    summary = dict(version='v4', n_cases=3301, length_min_words=64,
                                   groups={metric: dict.fromkeys(('live', 'non_live', 'multi_turn', 'average'), value)
                                           for metric in ('accuracy', 'format', 'length', 'length_ge_64', 'think_words')})
                    (out/'summary.json').write_text(json.dumps(summary))
                    (out/'inference.json').write_text(json.dumps(dict(temperature=profile['temperature'], top_p=1., length_min_words=64)))
            rows, methods = collect(manifest, root)
            self.assertEqual(len(rows), 4)
            self.assertEqual(len(methods), 2)
            cold = next(r for r in methods if r['temperature'] == .6)
            hot = next(r for r in methods if r['temperature'] == 1.)
            self.assertEqual(cold['completed_seeds'], 2)
            self.assertEqual(cold['seeds'], '1,2')
            self.assertAlmostEqual(cold['average_accuracy_mean'], 30)
            self.assertAlmostEqual(hot['average_accuracy_mean'], 15)
            self.assertAlmostEqual(cold['average_length_mean'], .3)
            self.assertAlmostEqual(cold['average_accuracy_sd'], 200**.5)


if __name__ == '__main__':
    unittest.main()
