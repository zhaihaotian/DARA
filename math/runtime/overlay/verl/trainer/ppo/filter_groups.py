"""DAPO dynamic sampling (M3) + deficit-sized later rounds (M11), for verl 0.7.0.dev.

GDPO Appendix E Table 7 sets `filter_groups.enable=TRUE, metric=seq_reward`. Checked all
three verl generations: 0.1, 0.3.1 (verl-agent fork) and 0.7.0.dev all lack it -- it is a
DAPO feature (arXiv:2503.14476), not upstream verl. So it is ported, not reused.

Difference from the verl 0.1 original: the reward channels arrive as per-sequence scalars
in `non_tensor_batch` (0.7's reward_extra_info mechanism) rather than as
`batch['token_level_scores_correctness']` (N,T) tensors. The filtering logic is verbatim.
"""
import math
from collections import defaultdict

import numpy as np
import torch

from verl import DataProto


def select_uid_groups(batch: DataProto, keep_uids):
    """Select whole uid groups, preserving group membership.

    Rows are chosen by uid, never by position slicing: GDPO's per-channel z-score and
    RD-GDPO's density pi_j are both computed per uid, so a partially-present group would
    silently corrupt both.
    """
    uids = batch.non_tensor_batch["uid"]
    keep = set(keep_uids)
    idx = np.array([i for i, u in enumerate(uids) if u in keep], dtype=np.int64)
    return DataProto(
        batch=batch.batch[torch.from_numpy(idx)],
        non_tensor_batch={k: v[idx] for k, v in batch.non_tensor_batch.items()},
        meta_info=batch.meta_info)


def filter_groups_by_reward(batch: DataProto, rollout_n: int, criterion="seq_reward"):
    """Drop groups whose rewards are all identical -- their in-group advantage is
    identically zero, so they contribute no gradient.

    criterion:
      'seq_reward'         the paper's literal metric: total sequence reward. DEFAULT.
      'any_channel_varies' keep if EITHER channel varies. An ABLATION, not the paper's
                           config; must never be reported as such.

    Why both exist: with two binary channels the totals collapse. (correct=1,length=0)
    and (correct=0,length=1) both sum to 1, so a group mixing them looks uniform under
    `seq_reward` while carrying the strongest correctness-vs-length competition -- exactly
    what GDPO section 4.2 studies. `reverse_conflict_rate` measures how often the two
    criteria disagree, so the choice is reported rather than silently made.

    Returns (keep_uids, uniform_uids, metrics).
    """
    uids = batch.non_tensor_batch["uid"]
    seq = batch.batch["token_level_scores"].sum(-1).float().cpu().numpy()

    # PORT: channels are per-sequence scalars in non_tensor_batch here, not (N,T) tensors
    has_ch = ("correctness" in batch.non_tensor_batch
              and "length" in batch.non_tensor_batch)
    if has_ch:
        ch_c = np.asarray(batch.non_tensor_batch["correctness"], dtype=np.float64)
        ch_l = np.asarray(batch.non_tensor_batch["length"], dtype=np.float64)

    uid2rows = defaultdict(list)
    for i, u in enumerate(uids):
        uid2rows[u].append(i)

    keep_uids, uniform_uids = [], []
    n_conflict = 0
    for u, rows in uid2rows.items():
        # a group must be complete, or grouping is broken upstream
        assert len(rows) == rollout_n, \
            f"uid {u} has {len(rows)} rows, expected rollout.n={rollout_n}"
        varies_seq = not np.all(seq[rows] == seq[rows][0])
        if has_ch:
            varies_ch = (not np.all(ch_c[rows] == ch_c[rows][0])) or \
                        (not np.all(ch_l[rows] == ch_l[rows][0]))
            if varies_ch and not varies_seq:
                n_conflict += 1          # channels compete but the totals cancel
        else:
            varies_ch = varies_seq

        keep = varies_ch if criterion == "any_channel_varies" else varies_seq
        (keep_uids if keep else uniform_uids).append(u)

    n_groups = len(uid2rows)
    metrics = {
        "filter/candidate_groups": float(n_groups),
        "filter/accepted_groups": float(len(keep_uids)),
        "filter/acceptance_rate": len(keep_uids) / max(n_groups, 1),
        "filter/reverse_conflict_groups": float(n_conflict),
        "filter/reverse_conflict_rate": n_conflict / max(n_groups, 1),
    }
    return keep_uids, uniform_uids, metrics


def deficit_request_size(deficit, acc_rate, world_size, n_prompts_full,
                         rollout_n, margin=1.25):
    """M11: size round >= 2 by the remaining deficit instead of regenerating a full batch.

    At an observed acceptance of ~0.70 a second full batch overshoots by ~40%. Asking for
    deficit/acc_rate (+25% margin) cuts generation by a measured 24-28%.

    MUST be a multiple of world_size: verl splits a DataProto across DP ranks and asserts
    divisibility in decorator.py. This actually killed a run -- deficit=147 at rate=0.713
    produced 258, and 258 % 8 == 2.
    """
    n = max(rollout_n, int(math.ceil(deficit / max(acc_rate, 0.05) * margin)))
    n = int(math.ceil(n / world_size) * world_size)
    return min(n_prompts_full, n)
