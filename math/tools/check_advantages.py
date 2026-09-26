"""Small CPU checks tied directly to the paper formulas and baseline contracts."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository math/ root
import torch
from rdgdpo.advantages import compute_advantages, standardize, group_rows


def main():
    torch.set_num_threads(1)
    uids = ["a", "b", "a", "b", "a", "b", "a", "b"]
    mask = torch.arange(9)[None, :] < torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])[:, None]
    rewards = {"correctness": [0., 0., 1., 1., 0., 0., 1., 1.],
               "length": [0., 0., 0., 1., 0., 0., 0., 1.]}
    gdpo, _, _ = compute_advantages("gdpo", rewards, uids, mask)
    capped, _, _ = compute_advantages("rdgdpo", rewards, uids, mask, {"w_max": 1.})
    assert torch.equal(gdpo, capped), "RD(w=1) must recover GDPO even with unequal lengths"
    rd, metrics, _ = compute_advantages("rdgdpo", rewards, uids, mask)
    assert metrics["rdgdpo/w_length"] > 1
    assert metrics["rdgdpo/energy_neg_length"] == metrics["support/energy_neg_length"]
    assert abs(metrics["rdgdpo/energy_pos_length"] - metrics["support/energy_pos_length"]*2) < 1e-5
    perm = torch.tensor([7, 1, 3, 2, 0, 5, 6, 4])
    permuted, _, _ = compute_advantages("rdgdpo", {k:torch.tensor(v)[perm] for k,v in rewards.items()},
                                       [uids[i] for i in perm], mask[perm])
    assert torch.allclose(permuted, rd[perm]), "UID grouping must survive row shuffling"
    longer, _, _ = compute_advantages("rdgdpo", rewards, uids, torch.ones(8, 25))
    assert torch.allclose(longer[:, 0], rd[:, 0]), "Response lengths must not reweight whitening"
    scalar = torch.stack([torch.tensor(v) for v in rewards.values()], -1)
    z = standardize(scalar, group_rows(uids))
    for rows in group_rows(uids):
        std = scalar[rows].std(0, correction=1)
        expected = 3*(std/(std+1e-6)).square()
        assert torch.allclose(z[rows].square().sum(0), expected, atol=1e-6)
    zero = {k:[0.]*8 for k in rewards}
    for method in ("grpo", "gdpo", "dvao", "gd2po_hard", "rdgdpo"):
        result, _, _ = compute_advantages(method, zero, uids, mask)
        assert torch.isfinite(result).all() and result.count_nonzero() == 0
        result, _, _ = compute_advantages(method, rewards, uids, mask)
        assert torch.isfinite(result).all() and result[~mask].count_nonzero() == 0
    opposite = {"correctness":[0.,1.,0.,1.], "length":[1.,0.,0.,1.]}
    hard, _, extra = compute_advantages("gd2po_hard", opposite, [0]*4, torch.ones(4, 3))
    assert hard[:2].count_nonzero() == 0
    assert torch.equal(extra["gd2po_query_weight"], torch.full((4,), .5))
    symmetric, _, _ = compute_advantages("rdgdpo", rewards, uids, mask, {"calibration":"symmetric"})
    assert not torch.allclose(symmetric, rd), "Two-sided DARA must differ from one-sided DARA"
    static_one, _, _ = compute_advantages("rdgdpo", rewards, uids, mask,
                                          {"static_weights": [1., 1.], "calibration": "symmetric"})
    assert torch.equal(static_one, gdpo), "Static weights (1, 1) must recover GDPO"
    static_rd, static_metrics, _ = compute_advantages(
        "rdgdpo", rewards, uids, mask, {"static_weights": [1., metrics["rdgdpo/w_length"]]})
    assert torch.allclose(static_rd, rd) and static_metrics["rdgdpo/w_length"] == metrics["rdgdpo/w_length"]
    # DVAO group population std denominator, independently calculated.
    d, _, _ = compute_advantages("dvao", opposite, [0]*4, torch.ones(4, 3))
    assert torch.allclose(d[:, 0], torch.tensor([0., 0., -1., 1.]))
    print("Checks passed: finite advantages for all methods, E/pi, one-sided energy, w=1 recovers GDPO, "
          "UID permutation, response weighting, GD2PO mask/loss weights, DVAO, DARA-Sym, Static weights.")


if __name__ == "__main__":
    main()
