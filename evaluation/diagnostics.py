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


def score_case(row, length_min_words=0):
    texts = [clean(t) for t in flatten(row['result'])]
    formats, lengths, compliance, words, binary_compliance = [], [], [], [], []
    with contextlib.redirect_stdout(io.StringIO()):
        for text in texts:
            completion = [[dict(role='assistant', content=text)]]
            formats.append(float(any(rlla.customize_format_reward_func(completion, [selector], 0, 1, 0)[0]
                                     for selector in ('<tool_call>', '<response>', '<tool_call><response>'))))
            lengths.append(rlla.customize_length_reward_func(
                completion, [''], 0, 1, 0, length_min_words=length_min_words)[0])
            valid = '<think>' in text and '</think>' in text
            n = len(text.split('<think>')[-1].split('</think>')[0].strip().split()) if valid else 0
            words.append(n)
            compliance.append(float(valid and n >= 512))
            binary_compliance.append(float(valid and n >= length_min_words))
    means = {name: statistics.mean(values) if values else 0.0
             for name, values in [('format', formats), ('length', lengths),
                                  ('length_ge_512', compliance), ('think_words', words)]}
    if length_min_words:
        means[f'length_ge_{length_min_words}'] = statistics.mean(binary_compliance) if binary_compliance else 0.0
    return dict(id=row['id'], emissions=len(texts), **means)
