"""Response-scalar advantage computation for GRPO, GDPO, DVAO, GD2PO-Hard and DARA.

GDPO-family final normalization is response-equal for every method in the
comparison. GD2PO query weights stay outside the clipped surrogate.
"""
from collections import defaultdict

import torch

METHODS = ("grpo", "gdpo", "dvao", "gd2po_hard", "rdgdpo")


def group_rows(index):
    groups = defaultdict(list)
    for row, uid in enumerate(index):
        groups[uid.item() if torch.is_tensor(uid) else uid].append(row)
    return list(groups.values())


def standardize(scores, groups, epsilon=1e-6):
    out = torch.zeros_like(scores)
    for rows in groups:
        values = scores[rows]
        if len(rows) < 2:
            raise ValueError("Each prompt needs at least two responses")
        out[rows] = (values-values.mean(dim=0)) / (values.std(dim=0, correction=1)+epsilon)
    return out


def normalize_responses(values, epsilon=1e-6, keep=None):
    if keep is None:
        keep = torch.ones_like(values, dtype=torch.bool)
    out = torch.zeros_like(values)
    if keep.sum() > 1:
        valid = values[keep]
        out[keep] = (valid-valid.mean()) / (valid.std(correction=1)+epsilon)
    return out


def support_statistics(channel_advantages, groups, names, tau=1e-8, prefix="support"):
    active = torch.stack([
        channel_advantages[rows].abs().sum(dim=0) > tau for rows in groups
    ])
    density = active.float().mean(dim=0)
    metrics = {f"{prefix}/n_groups": float(len(groups))}
    for k, name in enumerate(names):
        values = channel_advantages[:, k]
        metrics.update({
            f"{prefix}/pi_{name}": float(density[k]),
            f"{prefix}/active_groups_{name}": float(active[:, k].sum()),
            f"{prefix}/energy_pos_{name}": float(values.clamp(min=0).square().sum()),
            f"{prefix}/energy_neg_{name}": float(values.clamp(max=0).square().sum()),
        })
    return density, metrics


@torch.no_grad()
def compute_advantages(method, channels, index, response_mask, config=None):
    """Return (token advantages, scalar metrics, per-response loss weights).

    channels is an ordered dict: correctness plus exactly one of format/length.
    index groups responses by prompt UID, even when rows are not contiguous.
    The caller supplies the full update batch, before optimizer microbatching.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")
    names = tuple(channels)
    if names not in (("correctness", "format"), ("correctness", "length")):
        raise ValueError("Use explicit correctness/format or correctness/length channels")
    config = config or {}
    epsilon = float(config.get("epsilon", 1e-6))
    tau = float(config.get("tau", 1e-8))
    if epsilon <= 0 or tau < 0:
        raise ValueError("epsilon must be positive and tau non-negative")
    scores = torch.stack([torch.as_tensor(channels[n], device=response_mask.device,
                                         dtype=torch.float32) for n in names], dim=-1)
    if scores.shape != (len(index), len(names)) or response_mask.shape[0] != len(index):
        raise ValueError("Rewards, group IDs and response masks must have matching rows")
    if len(index) == 0 or not torch.isfinite(scores).all() or (response_mask.sum(-1) <= 0).any():
        raise ValueError("Need finite rewards and at least one valid token per response")
    groups = group_rows(index)
    channel_adv = standardize(scores, groups, epsilon)
    density, metrics = support_statistics(channel_adv, groups, names, tau)
    extra = {}

    if method == "grpo":
        result = standardize(scores.sum(-1, keepdim=True), groups, epsilon).squeeze(-1)
    elif method == "gdpo":
        result = normalize_responses(channel_adv.sum(-1), epsilon)
    elif method == "rdgdpo":
        w_max = float(config.get("w_max", 5.0))
        if w_max < 1:
            raise ValueError("w_max must be at least one")
        static = config.get("static_weights")
        if static is not None:
            # Static-Asym / Static-Sym controls: fixed channel weights instead of density weights.
            weights = scores.new_tensor([float(v) for v in static])
            if weights.shape != density.shape or (weights < 1).any():
                raise ValueError("static_weights needs one weight >= 1 per reward channel")
        else:
            weights = torch.ones_like(density)
            live = density > 0
            if live.any():
                weights[live] = torch.sqrt(density[live].max()/density[live]).clamp(max=w_max)
        calibration = config.get("calibration", "positive")
        if calibration == "positive":
            calibrated = weights*channel_adv.clamp(min=0)+channel_adv.clamp(max=0)
        elif calibration == "symmetric":
            calibrated = weights*channel_adv
        else:
            raise ValueError("calibration must be positive or symmetric")
        result = normalize_responses(calibrated.sum(-1), epsilon)
        for k, name in enumerate(names):
            metrics.update({
                f"rdgdpo/pi_{name}": float(density[k]),
                f"rdgdpo/w_{name}": float(weights[k]),
                f"rdgdpo/energy_pos_{name}": float(calibrated[:, k].clamp(min=0).square().sum()),
                f"rdgdpo/energy_neg_{name}": float(calibrated[:, k].clamp(max=0).square().sum()),
            })
    elif method == "dvao":
        weights = scores.new_tensor(config.get("base_weights", [0.5, 0.5]))
        if weights.shape != (len(names),) or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("DVAO needs non-negative channel weights with a positive sum")
        result = torch.zeros_like(scores[:, 0])
        for rows in groups:
            values = scores[rows]
            denominator = (weights*values.std(dim=0, correction=0)).sum()
            if denominator > float(config.get("denominator_epsilon", 1e-8)):
                result[rows] = ((values-values.mean(0))*weights).sum(-1)/denominator
    else:  # GD2PO-Hard: neutral zeros do not conflict.
        sign_epsilon = float(config.get("sign_epsilon", 1e-8))
        conflict = (channel_adv > sign_epsilon).any(-1) & (channel_adv < -sign_epsilon).any(-1)
        keep = ~conflict
        query_weight = torch.zeros_like(scores[:, 0])
        for rows in groups:
            query_weight[rows] = keep[rows].float().mean()
        result = normalize_responses(channel_adv.sum(-1), epsilon, keep)
        extra["gd2po_query_weight"] = query_weight
        metrics["gd2po_hard/conflict_ratio"] = float(conflict.float().mean())
        metrics["gd2po_hard/keep_ratio"] = float(keep.float().mean())
    return result.unsqueeze(-1)*response_mask, metrics, extra
