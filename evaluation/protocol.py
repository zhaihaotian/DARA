"""The fixed BFCL subsets and aggregation used for the reference tables."""
import statistics

V4_COMMIT = '6ea57973c7a6097fd7c5915698c54c17c5b1b6c8'
V3_PACKAGE = '2025.8.6.2'
CALLS = ['multiple', 'parallel', 'parallel_multiple']
LIVE = ['live_simple', 'live_multiple', 'live_parallel', 'live_parallel_multiple']
MULTI = ['multi_turn_base', 'multi_turn_miss_func', 'multi_turn_miss_param', 'multi_turn_long_context']
SIMPLE = {'v3': ['simple', 'java', 'javascript'],
          'v4': ['simple_python', 'simple_java', 'simple_javascript']}


def categories(version):
    return SIMPLE[version] + CALLS + LIVE + (MULTI if version == 'v4' else [])


def expected_counts(version):
    return dict(zip(categories(version), [400, 100, 50, 200, 200, 200, 258, 1053, 16, 24]
                    + ([200] * 4 if version == 'v4' else [])))


def aggregate(values, counts, version):
    """Input and output fractions: Simple and Non-Live macro; Live micro."""
    simple = statistics.mean(values[k] for k in SIMPLE[version])
    non_live = statistics.mean([simple] + [values[k] for k in CALLS])
    live = sum(values[k] * counts[k] for k in LIVE) / sum(counts[k] for k in LIVE)
    result = dict(non_live=non_live, live=live)
    if version == 'v4':
        multi = statistics.mean(values[k] for k in MULTI)
        result.update(multi_turn=multi, average=statistics.mean([non_live, live, multi]))
    return result
