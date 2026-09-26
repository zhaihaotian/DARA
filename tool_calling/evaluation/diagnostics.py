"""Compute per-case format and length scores from assistant outputs."""
import contextlib
import importlib.util
import io
from pathlib import Path
import statistics

SOURCE = Path(__file__).resolve().parents[1] / 'vendor/verl/verl/utils/reward_score/rlla.py'
spec = importlib.util.spec_from_file_location('rlla_diagnostics', SOURCE)
rlla = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rlla)


def flatten(value):
    if isinstance(value, str):
        return [value]
    return [text for item in value for text in flatten(item)]


def clean(text):
    return text.split('<|im_start|>assistant')[-1].split('<|im_end|>')[0].strip()


def score_case(row, length_max_words=0):
    """Format for every case; with length_max_words > 0 also the binary length reward and think words."""
    texts = [clean(t) for t in flatten(row['result'])]
    formats, lengths, words = [], [], []
    with contextlib.redirect_stdout(io.StringIO()):
        for text in texts:
            completion = [[dict(role='assistant', content=text)]]
            formats.append(float(any(rlla.customize_format_reward_func(completion, [selector], 0, 1, 0)[0]
                                     for selector in ('<tool_call>', '<response>', '<tool_call><response>'))))
            if length_max_words:
                lengths.append(rlla.customize_length_reward_func(
                    completion, [''], 0, 1, 0, length_max_words=length_max_words)[0])
                valid = '<think>' in text and '</think>' in text
                words.append(len(text.split('<think>')[-1].split('</think>')[0].strip().split()) if valid else 0)
    mean = lambda values: statistics.mean(values) if values else 0.0
    means = dict(format=mean(formats))
    if length_max_words:
        means.update({'length': mean(lengths), f'length_le_{length_max_words}': mean(lengths),
                      'think_words': mean(words)})
    return dict(id=row['id'], emissions=len(texts), **means)
