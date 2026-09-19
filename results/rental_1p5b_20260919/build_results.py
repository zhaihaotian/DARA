#!/usr/bin/env python3

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAINING_ROOT = ROOT / "training" / "runs"
EVALUATION_ROOT = ROOT / "evaluation"


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_training_exports():
    runs = []
    dynamics = []
    metric_columns = []

    for run_dir in sorted(path for path in TRAINING_ROOT.iterdir() if path.is_dir()):
        runtime = json.loads((run_dir / "runtime.json").read_text())
        manifest = runtime["manifest"]
        run_id = runtime["run_id"]
        method = manifest["method"]
        seed = manifest["seed"]
        model_size = manifest["model_size"]
        dynamics_rows = read_csv(run_dir / "training_dynamics.csv")
        metrics_rows = (run_dir / "metrics.jsonl").read_text().splitlines()
        if len(dynamics_rows) != 101 or len(metrics_rows) != 101:
            raise ValueError(f"{run_id}: expected 101 dynamics and metrics rows")

        for column in dynamics_rows[0]:
            if column not in metric_columns:
                metric_columns.append(column)
        for row in dynamics_rows:
            dynamics.append(
                {
                    "run_id": run_id,
                    "model_size": model_size,
                    "method": method,
                    "seed": seed,
                    **row,
                }
            )

        checkpoint_steps = sorted(
            int(path.name.removeprefix("global_step_"))
            for path in (run_dir / "actor").glob("global_step_*")
            if (path / "config.json").exists()
        )
        started_at = datetime.fromtimestamp(runtime["started_at"], timezone.utc).isoformat()
        runs.append(
            {
                "run_id": run_id,
                "model_size": model_size,
                "method": method,
                "seed": seed,
                "world_size": runtime["world_size"],
                "cuda_visible_devices": runtime["cuda_visible_devices"],
                "started_at_utc": started_at,
                "elapsed_seconds": (run_dir / "elapsed_seconds.txt").read_text().strip(),
                "metrics_rows": len(metrics_rows),
                "dynamics_rows": len(dynamics_rows),
                "saved_checkpoint_steps": "/".join(map(str, checkpoint_steps)),
            }
        )

    if len(runs) != 15:
        raise ValueError(f"expected 15 training runs, found {len(runs)}")
    write_csv(
        ROOT / "training" / "training_runs.csv",
        list(runs[0]),
        runs,
    )
    write_csv(
        ROOT / "training" / "training_dynamics.csv",
        ["run_id", "model_size", "method", "seed", *metric_columns],
        dynamics,
    )
    return runs, dynamics


def validate_evaluation_exports():
    checkpoint_rows = read_csv(EVALUATION_ROOT / "checkpoint_per_model.csv")
    checkpoint_methods = read_csv(EVALUATION_ROOT / "checkpoint_method_results.csv")
    final_rows = read_csv(EVALUATION_ROOT / "process_final_per_model.csv")
    final_methods = read_csv(EVALUATION_ROOT / "process_final_method_results.csv")

    checkpoint_keys = {(row["run_id"], int(row["step"])) for row in checkpoint_rows}
    if len(checkpoint_rows) != 150 or len(checkpoint_keys) != 150:
        raise ValueError("checkpoint_per_model.csv must contain 150 unique run-step rows")
    if {int(row["step"]) for row in checkpoint_rows} != set(range(10, 101, 10)):
        raise ValueError("checkpoint rows must cover steps 10 through 100")
    if len({row["run_id"] for row in checkpoint_rows}) != 15:
        raise ValueError("checkpoint rows must cover 15 runs")
    if len(checkpoint_methods) != 50:
        raise ValueError("checkpoint_method_results.csv must contain 5 methods x 10 steps")
    if any(row["completed_seeds"] != "3" or row["expected_seeds"] != "3" for row in checkpoint_methods):
        raise ValueError("every method-step aggregate must contain all three seeds")

    final_keys = {(row["run_id"], int(row["step"])) for row in final_rows}
    if len(final_rows) != 15 or len(final_keys) != 15:
        raise ValueError("process_final_per_model.csv must contain 15 unique step100 rows")
    if {int(row["step"]) for row in final_rows} != {100}:
        raise ValueError("final rows must all use step100")
    if len(final_methods) != 5:
        raise ValueError("process_final_method_results.csv must contain five methods")
    if any(row["completed_seeds"] != "3" or row["expected_seeds"] != "3" for row in final_methods):
        raise ValueError("every final aggregate must contain all three seeds")

    return checkpoint_rows, checkpoint_methods, final_rows, final_methods


def main():
    runs, dynamics = build_training_exports()
    checkpoint_rows, checkpoint_methods, final_rows, final_methods = validate_evaluation_exports()
    checkpoint_inventory = read_csv(ROOT / "training" / "checkpoint_inventory.csv")
    checkpoint_inventory_keys = {
        (row["run_id"], int(row["step"])) for row in checkpoint_inventory
    }
    if len(checkpoint_inventory) != 150 or len(checkpoint_inventory_keys) != 150:
        raise ValueError("checkpoint_inventory.csv must contain 150 unique run-step rows")
    raw_archives = read_csv(EVALUATION_ROOT / "bfcl_v4_archives.csv")
    if len(raw_archives) != 15 or len({row["archive"] for row in raw_archives}) != 15:
        raise ValueError("bfcl_v4_archives.csv must contain one archive per training run")
    verification = {
        "training_runs": len(runs),
        "training_dynamics_rows": len(dynamics),
        "saved_checkpoints": 150,
        "checkpoint_bytes": sum(int(row["size_bytes"]) for row in checkpoint_inventory),
        "checkpoint_evaluations": len(checkpoint_rows),
        "checkpoint_method_step_aggregates": len(checkpoint_methods),
        "final_evaluations": len(final_rows),
        "final_method_aggregates": len(final_methods),
        "bfcl_v4_raw_archives": len(raw_archives),
        "bfcl_v4_raw_archive_bytes": sum(int(row["size_bytes"]) for row in raw_archives),
        "all_training_rows_complete": all(row["metrics_rows"] == 101 for row in runs),
        "all_evaluation_keys_unique": len({(row["run_id"], row["step"]) for row in checkpoint_rows}) == 150,
        "all_checkpoint_inventory_keys_unique": len(checkpoint_inventory_keys) == 150,
        "controller_completion_time_utc": "2026-09-19T18:11:23.275409+00:00",
        "controller_completion_reason": "all_complete",
    }
    (ROOT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
