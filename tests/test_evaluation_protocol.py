from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'evaluation'))
from protocol import aggregate, expected_counts
from diagnostics import score_case


class EvaluationProtocolTest(unittest.TestCase):
    def test_live_is_micro_nonlive_languages_are_macro(self):
        counts = expected_counts('v4'); values = dict.fromkeys(counts, 0.)
        values.update(simple_java=1, live_parallel=1, multi_turn_base=1)
        result = aggregate(values, counts, 'v4')
        self.assertAlmostEqual(result['non_live'], 1/12)
        self.assertAlmostEqual(result['live'], 16/1351)
        self.assertAlmostEqual(result['multi_turn'], .25)
        self.assertAlmostEqual(result['average'], (1/12+16/1351+.25)/3)

    def test_multiturn_diagnostic_uses_case_equal_weight(self):
        valid = '<think>yes</think>\n<tool_call>\n[]\n</tool_call>'
        result = score_case({'id':'multi_turn_base_0', 'result':[[valid, 'bad'], [valid]]})
        self.assertAlmostEqual(result['format'], 2/3)
        self.assertEqual(result['emissions'], 3)

    def test_length_compliance_is_distinct_from_rounded_reward(self):
        result = score_case({'id':'simple_0', 'result':'<think>'+'x '*510+'</think>\n<response>x</response>'})
        self.assertEqual(result['length'], 1)
        self.assertEqual(result['length_ge_512'], 0)
        self.assertEqual(result['format'], 1)


if __name__ == '__main__':
    unittest.main()
