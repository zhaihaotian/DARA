"""Compatibility helpers for the paper rollout-correction integration."""


def _config_value(config, key, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def maybe_apply_rollout_correction(batch, rollout_corr_config=None, policy_loss_config=None):
    """Apply rollout-log-prob bypass mode or request old-log-prob recomputation."""
    if rollout_corr_config is None:
        return True

    bypass_mode = _config_value(rollout_corr_config, "bypass_mode", None)
    if bypass_mode is None:
        bypass_mode = _config_value(
            rollout_corr_config, "bypass_old_logprob_for_rollout", False
        )
    if not bypass_mode:
        return True

    from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

    apply_bypass_mode(
        batch=batch,
        rollout_corr_config=rollout_corr_config,
        policy_loss_config=policy_loss_config,
    )
    return False
