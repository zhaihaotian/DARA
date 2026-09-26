# Advantage estimators

GRPO, GDPO and DARA are implemented in `vendor/verl/verl/trainer/ppo/core_algos.py`; DARA's combination is `compute_dara_combined_advantage`. The per-channel wiring is in `ray_trainer.py::compute_advantage`. DVAO and GD²PO-Hard live in `multi_reward_algos.py`. GD²PO-Hard's per-prompt weight is also applied inside the actor's PPO loss.

Notation: $Z_{ijk}$ is the group z-score of reward $k$ for response $j$ of prompt $i$ (the paper's $A_k^{(i,j)}$). It uses the sample standard deviation with denominator $\mathrm{std}+10^{-6}$. "Token whitening" is the final batch normalization over all valid response tokens: subtract the mean, divide by $\sqrt{\mathrm{var}+10^{-8}}$.

| Method | `--method` | Advantage |
|---|---|---|
| GRPO | `grpo` | Sum the channel rewards (after the reward-KL penalty), then take the group z-score. |
| GDPO | `gdpo` | $\sum_k Z_{ijk}$, then token whitening. |
| DARA-Asym | `dara_asym` | Density $\pi_k$ = fraction of prompt groups with a nonzero channel-$k$ advantage (summed absolute token advantage above $10^{-8}$); $w_k=\min(5,\sqrt{\max_l \pi_l/\pi_k})$. Only positive advantages are amplified: $\sum_k (Z_{ijk} + (w_k-1)[Z_{ijk}]_+)$, then token whitening. Main method. |
| DARA-Sym | `dara_sym` | Same $\pi_k$ and $w_k$; both signs are scaled: $\sum_k w_k Z_{ijk}$, then token whitening. |
| DVAO | `dvao` | Channel weights 0.5/0.5. The weighted centered reward is divided by the sum of weighted population standard deviations ($\epsilon=10^{-8}$). There is no final batch whitening. |
| GD²PO-Hard | `gd2po_hard` | A response whose channel advantages disagree in sign is masked (zero counts as neutral). The remaining advantages get token whitening. Each prompt's retained fraction multiplies the clipped PPO surrogate. |

Both DARA variants run through the same estimator (`dara`, selected by `algorithm.dara.calibration`: `positive` or `symmetric`). For a channel with $\pi_k=0$, the logged DARA weight is the cap, but its advantages are all zero, so the weight has no effect. With the same rewards, masks and group ids, DARA with all $w_k=1$ is exactly GDPO; the CPU tests check this, as well as each baseline against an independent NumPy reference.

## Logged diagnostics

GRPO, GDPO and DARA share `core_algos.compute_channel_densities`, which computes the per-channel density and active-group count from the raw-reward group z-scores. Each step logs `<method>/pi_<channel>`, `<method>/w_<channel>` and `<method>/active_groups_<channel>` (for GRPO and GDPO the recorded weight is 1). For GRPO these statistics are diagnostic only; its policy advantage still comes from the summed reward.
