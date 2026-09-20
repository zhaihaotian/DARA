#!/usr/bin/env python3

import argparse
import csv
import json
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAINING_ROOT = ROOT / "training" / "runs"
EVALUATION_ROOT = ROOT / "evaluation"
RUN_IDS = [
    "haotian_3b_dvao_s1_save10",
    "haotian_3b_dvao_s2_save10",
    "haotian_3b_gd2po_hard_s1_save10",
    "haotian_3b_gd2po_hard_s2_save10",
]
EXPECTED_STEPS = list(range(10, 101, 10))
EVALUATION_METRICS = [
    "live_accuracy",
    "live_format",
    "non_live_accuracy",
    "non_live_format",
    "multi_turn_accuracy",
    "multi_turn_format",
    "average_accuracy",
    "average_format",
]
REMOTE_INVENTORY_SCRIPT = r'''
import json
import os
import sys

root = sys.argv[1]
runs = sys.argv[2:]
rows = []
for run_id in runs:
    for step in range(10, 101, 10):
        checkpoint = os.path.join(
            root, "outputs", "training", run_id, "actor", f"global_step_{step}"
        )
        size_bytes = 0
        file_count = 0
        for directory, _, files in os.walk(checkpoint):
            for name in files:
                path = os.path.join(directory, name)
                size_bytes += os.path.getsize(path)
                file_count += 1
        rows.append(
            {
                "run_id": run_id,
                "step": step,
                "source_path": checkpoint,
                "size_bytes": size_bytes,
                "file_count": file_count,
            }
        )
print(json.dumps(rows))
'''


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def refresh_checkpoint_inventory(ssh_host, remote_root):
    completed = subprocess.run(
        ["ssh", ssh_host, "python3", "-", remote_root, *RUN_IDS],
        input=REMOTE_INVENTORY_SCRIPT,
        text=True,
        capture_output=True,
        check=True,
    )
    raw_rows = json.loads(completed.stdout)
    manifests = {}
    for run_id in RUN_IDS:
        runtime = json.loads((TRAINING_ROOT / run_id / "runtime.json").read_text())
        manifests[run_id] = runtime["manifest"]
    rows = []
    for row in raw_rows:
        manifest = manifests[row["run_id"]]
        rows.append(
            {
                "run_id": row["run_id"],
                "model_size": manifest["model_size"],
                "method": manifest["method"],
                "seed": manifest["seed"],
                "step": row["step"],
                "source_path": row["source_path"],
                "size_bytes": row["size_bytes"],
                "file_count": row["file_count"],
            }
        )
    write_csv(ROOT / "training" / "checkpoint_inventory.csv", list(rows[0]), rows)


def build_training_exports():
    runs = []
    dynamics = []
    metric_columns = []

    run_dirs = sorted(path for path in TRAINING_ROOT.iterdir() if path.is_dir())
    if {path.name for path in run_dirs} != set(RUN_IDS):
        raise ValueError("training/runs must contain exactly the four expected runs")

    for run_dir in run_dirs:
        runtime = json.loads((run_dir / "runtime.json").read_text())
        manifest = runtime["manifest"]
        run_id = runtime["run_id"]
        dynamics_rows = read_csv(run_dir / "training_dynamics.csv")
        metrics_lines = (run_dir / "metrics.jsonl").read_text().splitlines()
        metrics_rows = [json.loads(line) for line in metrics_lines]
        if len(dynamics_rows) != 101 or len(metrics_rows) != 101:
            raise ValueError(f"{run_id}: expected 101 dynamics and metrics rows")
        if [int(row["step"]) for row in dynamics_rows] != list(range(101)):
            raise ValueError(f"{run_id}: dynamics must cover steps 0 through 100")
        if [int(row["step"]) for row in metrics_rows] != list(range(101)):
            raise ValueError(f"{run_id}: metrics must cover steps 0 through 100")
        for metrics_row, dynamics_row in zip(metrics_rows, dynamics_rows):
            populated_dynamics = {key for key, value in dynamics_row.items() if value != ""}
            if populated_dynamics != set(metrics_row):
                raise ValueError(
                    f"{run_id} step {metrics_row['step']}: metrics/dynamics fields differ"
                )
            for key, value in metrics_row.items():
                if key == "step":
                    same_value = int(dynamics_row[key]) == int(value)
                elif isinstance(value, (int, float)):
                    same_value = float(dynamics_row[key]) == float(value)
                else:
                    same_value = dynamics_row[key] == str(value)
                if not same_value:
                    raise ValueError(
                        f"{run_id} step {metrics_row['step']}: {key} differs"
                    )

        for column in dynamics_rows[0]:
            if column not in metric_columns:
                metric_columns.append(column)
        for row in dynamics_rows:
            dynamics.append(
                {
                    "run_id": run_id,
                    "model_size": manifest["model_size"],
                    "method": manifest["method"],
                    "seed": manifest["seed"],
                    **row,
                }
            )

        checkpoint_steps = sorted(
            int(path.name.removeprefix("global_step_"))
            for path in (run_dir / "actor").glob("global_step_*")
            if (path / "config.json").exists()
        )
        if checkpoint_steps != EXPECTED_STEPS:
            raise ValueError(f"{run_id}: expected checkpoint configs at steps 10 through 100")
        started_at = datetime.fromtimestamp(runtime["started_at"], timezone.utc).isoformat()
        runs.append(
            {
                "run_id": run_id,
                "model_size": manifest["model_size"],
                "method": manifest["method"],
                "seed": manifest["seed"],
                "world_size": runtime["world_size"],
                "cuda_visible_devices": runtime["cuda_visible_devices"],
                "started_at_utc": started_at,
                "elapsed_seconds": (run_dir / "elapsed_seconds.txt").read_text().strip(),
                "metrics_rows": len(metrics_rows),
                "dynamics_rows": len(dynamics_rows),
                "saved_checkpoint_steps": "/".join(map(str, checkpoint_steps)),
            }
        )

    write_csv(ROOT / "training" / "training_runs.csv", list(runs[0]), runs)
    write_csv(
        ROOT / "training" / "training_dynamics.csv",
        ["run_id", "model_size", "method", "seed", *metric_columns],
        dynamics,
    )
    return runs, dynamics


def validate_aggregates(per_model_rows, aggregate_rows):
    grouped = {}
    for row in per_model_rows:
        grouped.setdefault((row["method"], int(row["step"])), []).append(row)
    aggregate_keys = {(row["method"], int(row["step"])) for row in aggregate_rows}
    if aggregate_keys != set(grouped):
        raise ValueError("method-step aggregate keys do not match per-run evaluation keys")
    for aggregate in aggregate_rows:
        key = (aggregate["method"], int(aggregate["step"]))
        source_rows = grouped[key]
        source_seeds = {int(row["seed"]) for row in source_rows}
        aggregate_seeds = {int(seed) for seed in aggregate["seeds"].split(",")}
        if len(source_rows) != 2 or source_seeds != aggregate_seeds:
            raise ValueError(f"{key}: aggregate seed membership differs")
        for metric in EVALUATION_METRICS:
            values = [float(row[metric]) for row in source_rows]
            expected_mean = statistics.mean(values)
            expected_sd = statistics.stdev(values)
            if not math.isclose(
                float(aggregate[f"{metric}_mean"]), expected_mean, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(f"{key}: {metric} mean differs")
            if not math.isclose(
                float(aggregate[f"{metric}_sd"]), expected_sd, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(f"{key}: {metric} sample standard deviation differs")


def validate_evaluation_exports():
    checkpoint_rows = read_csv(EVALUATION_ROOT / "checkpoint_per_model.csv")
    checkpoint_methods = read_csv(EVALUATION_ROOT / "checkpoint_method_results.csv")
    final_rows = read_csv(EVALUATION_ROOT / "process_final_per_model.csv")
    final_methods = read_csv(EVALUATION_ROOT / "process_final_method_results.csv")

    checkpoint_keys = {(row["run_id"], int(row["step"])) for row in checkpoint_rows}
    if len(checkpoint_rows) != 40 or len(checkpoint_keys) != 40:
        raise ValueError("checkpoint_per_model.csv must contain 40 unique run-step rows")
    if {int(row["step"]) for row in checkpoint_rows} != set(EXPECTED_STEPS):
        raise ValueError("checkpoint rows must cover steps 10 through 100")
    if {row["run_id"] for row in checkpoint_rows} != set(RUN_IDS):
        raise ValueError("checkpoint rows must cover the four expected runs")
    if len(checkpoint_methods) != 20:
        raise ValueError("checkpoint_method_results.csv must contain 2 methods x 10 steps")
    if any(row["completed_seeds"] != "2" or row["expected_seeds"] != "2" for row in checkpoint_methods):
        raise ValueError("every method-step aggregate must contain both seeds")
    validate_aggregates(checkpoint_rows, checkpoint_methods)

    final_keys = {(row["run_id"], int(row["step"])) for row in final_rows}
    if len(final_rows) != 4 or len(final_keys) != 4:
        raise ValueError("process_final_per_model.csv must contain four unique step100 rows")
    if {int(row["step"]) for row in final_rows} != {100}:
        raise ValueError("final rows must all use step100")
    if len(final_methods) != 2:
        raise ValueError("process_final_method_results.csv must contain two methods")
    if any(row["completed_seeds"] != "2" or row["expected_seeds"] != "2" for row in final_methods):
        raise ValueError("every final aggregate must contain both seeds")
    validate_aggregates(final_rows, final_methods)
    checkpoint_step100 = {
        row["run_id"]: row for row in checkpoint_rows if int(row["step"]) == 100
    }
    for final_row in final_rows:
        checkpoint_row = checkpoint_step100[final_row["run_id"]]
        if any(final_row[field] != checkpoint_row[field] for field in final_row):
            raise ValueError(f"{final_row['run_id']}: final row differs from checkpoint step100")

    return checkpoint_rows, checkpoint_methods, final_rows, final_methods


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-inventory-host")
    parser.add_argument(
        "--remote-root", default="/lambda/nfs/haotian/dara-3b-20260919"
    )
    args = parser.parse_args()
    if args.refresh_inventory_host:
        refresh_checkpoint_inventory(args.refresh_inventory_host, args.remote_root)

    runs, dynamics = build_training_exports()
    checkpoint_rows, checkpoint_methods, final_rows, final_methods = validate_evaluation_exports()
    checkpoint_inventory = read_csv(ROOT / "training" / "checkpoint_inventory.csv")
    checkpoint_inventory_keys = {
        (row["run_id"], int(row["step"])) for row in checkpoint_inventory
    }
    if len(checkpoint_inventory) != 40 or len(checkpoint_inventory_keys) != 40:
        raise ValueError("checkpoint_inventory.csv must contain 40 unique run-step rows")
    if any(int(row["size_bytes"]) <= 0 or int(row["file_count"]) <= 0 for row in checkpoint_inventory):
        raise ValueError("every checkpoint inventory row must have a positive size and file count")

    status = json.loads((ROOT / "operations" / "final_status.json").read_text())
    queue = json.loads((ROOT / "operations" / "queue_state.json").read_text())
    if not (
        status["training_complete"] == status["training_expected"] == 4
        and status["checkpoints_scored"] == status["checkpoints_expected"] == 40
        and status["finals_scored"] == status["finals_expected"] == 4
        and not status["active"]
        and not status["training_failed"]
        and not status["evaluations_remaining"]
    ):
        raise ValueError("operations/final_status.json is not a complete final state")

    metadata_names = {
        "metrics.jsonl",
        "training_dynamics.csv",
        "runtime.json",
        "config.json",
        "command.json",
        "launch.json",
        "elapsed_seconds.txt",
        "exit.code",
        "worker.exit.code",
    }
    all_run_metadata_complete = all(
        metadata_names.issubset({path.name for path in (TRAINING_ROOT / run_id).iterdir()})
        for run_id in RUN_IDS
    )
    verification = {
        "training_runs": len(runs),
        "training_dynamics_rows": len(dynamics),
        "saved_checkpoints": len(checkpoint_inventory),
        "checkpoint_bytes": sum(int(row["size_bytes"]) for row in checkpoint_inventory),
        "checkpoint_evaluations": len(checkpoint_rows),
        "checkpoint_method_step_aggregates": len(checkpoint_methods),
        "final_evaluations": len(final_rows),
        "final_method_aggregates": len(final_methods),
        "checkpoint_config_files": sum(
            1 for path in TRAINING_ROOT.glob("*/actor/global_step_*/config.json")
        ),
        "all_training_rows_complete": all(row["metrics_rows"] == 101 for row in runs),
        "all_evaluation_keys_unique": len(
            {(row["run_id"], row["step"]) for row in checkpoint_rows}
        ) == 40,
        "all_checkpoint_inventory_keys_unique": len(checkpoint_inventory_keys) == 40,
        "all_run_metadata_complete": all_run_metadata_complete,
        "controller_completion_time_utc": queue["updated_at"],
        "controller_completion_reason": "all_complete",
    }
    (ROOT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
