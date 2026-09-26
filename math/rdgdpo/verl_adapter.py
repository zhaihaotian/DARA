"""Single verl 0.7 adapter installed as verl.trainer.ppo.gdpo_estimators."""
import torch
from verl.trainer.ppo.core_algos import register_adv_est
from rdgdpo.advantages import compute_advantages, standardize, group_rows, support_statistics

LAST_RDGDPO_METRICS = {}
LAST_MAIN_COMPARISON_METRICS = {}


def _channels(correctness=None, format=None, length=None):
    if correctness is None or (format is None) == (length is None):
        raise ValueError("Provide correctness plus exactly one of format or length")
    return {"correctness": correctness, **({"format": format} if format is not None else {"length": length})}


def _run(method, response_mask, index, config, correctness, format, length):
    channels = _channels(correctness, format, length)
    if config is None or method not in config:
        raise KeyError(f"Missing algorithm.{method} configuration")
    cfg = dict(config[method])
    # Epsilon for group/response normalization is shared by GDPO-family methods.
    cfg["epsilon"] = 1e-6
    if method == "dvao" and "epsilon" in (config or {}).get(method, {}):
        cfg["denominator_epsilon"] = config[method]["epsilon"]
    advantages, metrics, extra = compute_advantages(method, channels, index, response_mask, cfg)
    global LAST_MAIN_COMPARISON_METRICS
    LAST_MAIN_COMPARISON_METRICS = metrics
    return (advantages, advantages, extra) if extra else (advantages, advantages)


def _estimator(method):
    def estimate(token_level_rewards, response_mask, index=None, config=None,
                 correctness=None, format=None, length=None, **kwargs):
        return _run(method, response_mask, index, config, correctness, format, length)
    estimate.__name__ = method + "_adv_est"
    return register_adv_est(method)(estimate)


gdpo_adv_est = _estimator("gdpo")
rdgdpo_adv_est = _estimator("rdgdpo")
dvao_adv_est = _estimator("dvao")
gd2po_hard_adv_est = _estimator("gd2po_hard")


def batch_support_metrics(batch, prefix="support"):
    """Works for GRPO too, before optimizer microbatching or group filtering."""
    data = batch.non_tensor_batch
    channels = _channels(data.get("correctness"), data.get("format"), data.get("length"))
    scores = torch.stack([torch.as_tensor(v, dtype=torch.float32) for v in channels.values()], -1)
    groups = group_rows(data["uid"])
    adv = standardize(scores, groups)
    _, metrics = support_statistics(adv, groups, tuple(channels), prefix=prefix)
    return metrics
