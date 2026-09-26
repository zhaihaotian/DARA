"""Calculate sustained reward crossing from one run's prefilter CSV (stdlib only)."""
import argparse
import csv
import json
import math
from pathlib import Path


def sustained_crossing(points, threshold=0.99, window=10, future=20):
    """Return first crossing and confirmation updates; never fill missing observations.

    points: ascending or unsorted (positive integer update, reward in [0,1]) pairs.
    An incomplete/missing history cannot establish the *first* crossing and is rejected.
    A valid observed prefix with no confirmed crossing is right-censored.
    """
    if not 0 <= threshold <= 1 or window < 1 or future < 0:
        raise ValueError("threshold must be in [0,1], window>=1 and future>=0")
    points = sorted(points)
    if not points:
        raise ValueError("No observed updates")
    steps = [p[0] for p in points]
    if steps != list(range(1, len(points) + 1)):
        raise ValueError("Need every update from 1, exactly once; do not fill gaps")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for _, v in points):
        raise ValueError("Reward must be finite and in [0,1]")
    rewards = [p[1] for p in points]
    moving = {
        t: math.fsum(rewards[t-window:t]) / window
        for t in range(window, len(points) + 1)
    }
    candidates = [t for t in moving if t + future <= len(points)]
    crossing = next((t for t in candidates
                     if all(moving[u] >= threshold for u in range(t, t+future+1))), None)
    pending = None
    if crossing is None:
        pending = next((t for t in moving if t+future > len(points)
                        and all(moving[u] >= threshold for u in range(t, len(points)+1))), None)
    return {
        "threshold": threshold, "window": window, "future_points": future,
        "observed_updates": len(points),
        "status": "reached" if crossing is not None else "not_confirmed",
        "crossing_update": crossing,
        "confirmation_update": crossing + future if crossing is not None else None,
        "pending_crossing_update": pending,
        "last_confirmable_update": max(candidates) if candidates else None,
    }


def read_points(path, step_column, metric, budget):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not {step_column, metric}.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV needs {step_column!r} and {metric!r}")
        points = []
        for row in reader:
            step_value = float(row[step_column])
            if not step_value.is_integer():
                raise ValueError("Update must be an integer")
            step = int(step_value)
            if step == 0 or step > budget:
                continue
            points.append((step, float(row[metric])))
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    parser.add_argument("--label", required=True, help="Include method variant and seed")
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--step-column", default="step")
    parser.add_argument("--metric", default="rollout_prefilter/length/mean")
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--future", type=int, default=20)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.budget < 1:
        parser.error("budget must be positive")
    try:
        result = sustained_crossing(
            read_points(args.csv, args.step_column, args.metric, args.budget),
            args.threshold, args.window, args.future)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    result.update(label=args.label, source=str(Path(args.csv).resolve()), metric=args.metric,
                  requested_budget=args.budget,
                  budget_complete=result["observed_updates"] == args.budget)
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
