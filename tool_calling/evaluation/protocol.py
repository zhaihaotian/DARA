"""BFCL V4 subsets and the group aggregation used for evaluation."""
import statistics

V4_COMMIT = '6ea57973c7a6097fd7c5915698c54c17c5b1b6c8'
SIMPLE = ['simple_python', 'simple_java', 'simple_javascript']
CALLS = ['multiple', 'parallel', 'parallel_multiple']
LIVE = ['live_simple', 'live_multiple', 'live_parallel', 'live_parallel_multiple']
MULTI = ['multi_turn_base', 'multi_turn_miss_func', 'multi_turn_miss_param', 'multi_turn_long_context']


def categories():
    return SIMPLE + CALLS + LIVE + MULTI


def expected_counts():
    return dict(zip(categories(), [400, 100, 50, 200, 200, 200, 258, 1053, 16, 24] + [200] * 4))


def aggregate(values, counts):
    """Input and output fractions: Simple and Non-Live macro; Live micro; Multi-Turn macro."""
    simple = statistics.mean(values[k] for k in SIMPLE)
    non_live = statistics.mean([simple] + [values[k] for k in CALLS])
    live = sum(values[k] * counts[k] for k in LIVE) / sum(counts[k] for k in LIVE)
    multi = statistics.mean(values[k] for k in MULTI)
    return dict(non_live=non_live, live=live, multi_turn=multi,
                average=statistics.mean([non_live, live, multi]))
