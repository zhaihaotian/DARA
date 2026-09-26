"""Additional Tool Calling estimators on the existing GDPO training path.

Channel standardization and token-weighted batch whitening use the same
functions as this repository's GDPO estimator. DVAO has no batch whitening.
"""

from collections import defaultdict

import torch

from verl.trainer.ppo import core_algos
from verl.utils.torch_functional import masked_whiten


METHODS = ("dvao", "gd2po_hard")


@torch.no_grad()
def compute_multi_reward_advantage(method, correctness, format_reward,
                                   response_mask, index, config=None):
    """Return token advantages, scalar metrics, and optional actor loss weights."""
    if method not in METHODS:
        raise ValueError(f"Unknown additional reward estimator: {method}")
    config = config or {}
    channels = (correctness, format_reward)
    names = ("correctness", "format")
    scores = torch.stack([channel.sum(-1) for channel in channels], dim=-1)
    groups = defaultdict(list)
    for row, uid in enumerate(index):
        groups[uid].append(row)
    metrics, extra = {}, {}

    if method == "dvao":
        weights = scores.new_tensor(config.get("base_weights", [0.5, 0.5]))
        if weights.shape != (2,) or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("DVAO needs two nonnegative weights with positive sum")
        scalar_advantage = torch.zeros_like(scores[:, 0])
        for rows in groups.values():
            values = scores[rows]
            denominator = (weights * values.std(dim=0, unbiased=False)).sum()
            if denominator > float(config.get("denominator_epsilon", 1e-8)):
                scalar_advantage[rows] = ((values - values.mean(0)) * weights).sum(-1) / denominator
        return scalar_advantage.unsqueeze(-1) * response_mask, metrics, extra

    # Reuse the original per-channel GDPO calculation, including std + 1e-6.
    channel_advantages = torch.stack([
        core_algos.compute_grpo_outcome_advantage(channel, response_mask, index)[0]
        for channel in channels
    ], dim=-1)

    # GD2PO-Hard: responses whose channel advantages disagree in sign are masked.
    epsilon = float(config.get("sign_epsilon", 1e-8))
    positive = (channel_advantages > epsilon).any(dim=(1, 2))
    negative = (channel_advantages < -epsilon).any(dim=(1, 2))
    conflict = positive & negative
    keep = ~conflict
    keep_mask = response_mask * keep.unsqueeze(-1)
    combined = channel_advantages.sum(-1)
    # A batch with all responses filtered is a valid zero policy update.
    if keep_mask.sum() > 1:
        advantages = masked_whiten(combined, keep_mask) * keep_mask
    else:
        advantages = torch.zeros_like(combined)
    query_weight = torch.zeros_like(scores[:, 0])
    for rows in groups.values():
        query_weight[rows] = keep[rows].float().mean()
    extra["gd2po_query_weight"] = query_weight
    metrics["gd2po_hard/conflict_ratio"] = float(conflict.float().mean())
    metrics["gd2po_hard/keep_ratio"] = float(keep.float().mean())
    return advantages, metrics, extra
