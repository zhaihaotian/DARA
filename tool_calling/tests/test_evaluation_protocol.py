from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'evaluation'))
from protocol import aggregate, expected_counts
from diagnostics import score_case


class EvaluationProtocolTest(unittest.TestCase):
    def test_live_is_micro_nonlive_languages_are_macro(self):
        counts = expected_counts(); values = dict.fromkeys(counts, 0.)
        values.update(simple_java=1, live_parallel=1, multi_turn_base=1)
        result = aggregate(values, counts)
        self.assertAlmostEqual(result['non_live'], 1/12)
        self.assertAlmostEqual(result['live'], 16/1351)
        self.assertAlmostEqual(result['multi_turn'], .25)
        self.assertAlmostEqual(result['average'], (1/12+16/1351+.25)/3)

    def test_multiturn_diagnostic_uses_case_equal_weight(self):
        valid = '<think>yes</think>\n<tool_call>\n[]\n</tool_call>'
        result = score_case({'id':'multi_turn_base_0', 'result':[[valid, 'bad'], [valid]]})
        self.assertAlmostEqual(result['format'], 2/3)
        self.assertEqual(result['emissions'], 3)

    def test_binary_length_is_scored_only_when_requested(self):
        text = '<think>'+'x '*16+'</think>\n<response>x</response>'
        plain = score_case({'id':'simple_0', 'result':text})
        self.assertEqual(set(plain), {'id', 'emissions', 'format'})
        result = score_case({'id':'simple_0', 'result':text}, length_max_words=16)
        self.assertEqual(result['length'], 1)
        self.assertEqual(result['length_le_16'], 1)
        self.assertEqual(result['think_words'], 16)
        self.assertEqual(result['format'], 1)

if __name__ == '__main__':
    unittest.main()
