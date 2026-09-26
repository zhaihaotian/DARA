"""math scorer for verl 0.3.1 -- returns a dict so reward_extra_info carries both channels.

verl 0.1 hard-coded a 4-tuple return (score, format, correctness, length) all the way
through RewardManager. 0.3.1 has a cleaner extension point: if compute_score returns a
dict, every key lands in `reward_extra_info` and from there in
`batch.non_tensor_batch[key]` as one scalar per sequence. So multi-channel reward needs
NO architectural change here -- only a scorer that returns a dict.

Channel mapping (identical to the 0.1 track, see MASTER_HANDOFF 2.4):
    correctness  boxed-answer equivalence, {0,1}
    length       response <= 4000 tokens, {0,1}   <- the paper's R_length
The 0.1 code carried this in the "format" slot because the tool-calling track owned the
name; here it is called `length` outright, since nothing forces the old slot names.
`score` (their sum) stays the primary reward so single-channel estimators still work.
"""
import os

# verl 0.7 renamed reward_score/math.py -> math_reward.py. Verified same origin
# (hendrycks MATH grader, identical EleutherAI/HuggingFace headers) and `compute_score`
# is logically IDENTICAL -- the only diffs are black formatting plus one broadened
# `except AssertionError` -> `except Exception` in the string normaliser, which makes 0.7
# strictly more tolerant of malformed answers. Recorded as a deviation; it can only
# change verdicts on inputs that would previously have crashed the grader.
try:
    from verl.utils.reward_score import math_reward as math_scorer   # verl >= 0.7
except ImportError:                                                  # verl 0.1
    from verl.utils.reward_score import math as math_scorer


def _extract_response(solution_str):
    """Strip the prompt echo. R1-Distill's template pre-fills <think>, so the model's
    output begins after it; the 0.1 scorer did the same split."""
    if "<｜Assistant｜>" in solution_str:
        return solution_str.split("<｜Assistant｜>")[-1]
    return solution_str


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    predict_str = _extract_response(solution_str)

    correctness = float(math_scorer.compute_score(predict_str, ground_truth))

    # Token count must match what the trainer sees. extra_info carries the true
    # valid_response_length; falling back to a character heuristic would shift the
    # 4000-token boundary and silently change R_length (fix M5 in the 0.1 track).
    limit = int(os.getenv("LENGTH_LIMIT", 4000))
    n_tok = None
    if isinstance(extra_info, dict):
        n_tok = extra_info.get("valid_response_length")
    if n_tok is None:
        raise ValueError("Math reward requires extra_info.valid_response_length from the reward manager")
    length = float(int(n_tok) <= limit)

    return {
        "score": correctness + length,   # primary scalar reward
        "correctness": correctness,
        "length": length,
    }
