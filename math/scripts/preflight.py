"""Validate the frozen text-only RD-GDPO runtime before launching training."""

import argparse
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import sys


EXPECTED_VERSIONS = {
    "torch": "2.8.0",
    "vllm": "0.11.0",
    "transformers": "4.57.3",
    "ray": "2.50.0",
    "flash_attn": "2.8.3",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", type=Path, required=True)
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--min-gpus", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    framework = args.framework.expanduser().resolve()
    marker_path = framework / "rdgdpo_runtime.json"
    if not marker_path.is_file():
        raise SystemExit(f"Missing prepared runtime marker: {marker_path}")
    marker = json.loads(marker_path.read_text())
    if marker.get("profile") != "dara-math":
        raise SystemExit(f"Unexpected runtime profile: {marker.get('profile')}")

    sys.path.insert(0, str(framework))
    installed = {name: version(name) for name in EXPECTED_VERSIONS}
    mismatches = {
        name: {"expected": expected, "actual": installed[name]}
        for name, expected in EXPECTED_VERSIONS.items()
        if installed[name] != expected
    }
    if mismatches:
        raise SystemExit(f"Version mismatch: {json.dumps(mismatches, sort_keys=True)}")

    importlib.import_module("vllm._C")
    from rdgdpo.sync_dataset import sync_text_getitem
    from verl.trainer.ppo.rollout_corr_helper import maybe_apply_rollout_correction

    if not callable(sync_text_getitem) or not callable(maybe_apply_rollout_correction):
        raise SystemExit("Prepared runtime compatibility APIs are not callable")

    result = {
        "framework": str(framework),
        "versions": installed,
        "vllm_extension": "ok",
        "text_dataset_adapter": "ok",
        "rollout_correction_compat": "ok",
    }
    if not args.skip_gpu:
        import torch
        from flash_attn import flash_attn_func

        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available")
        if torch.cuda.device_count() < args.min_gpus:
            raise SystemExit(
                f"Need at least {args.min_gpus} GPUs, found {torch.cuda.device_count()}"
            )
        device = torch.device("cuda:0")
        matrix = torch.randn(32, 32, device=device, dtype=torch.bfloat16)
        product = matrix @ matrix
        query = torch.randn(1, 4, 2, 64, device=device, dtype=torch.bfloat16)
        attention = flash_attn_func(query, query, query)
        if not torch.isfinite(product).all() or not torch.isfinite(attention).all():
            raise SystemExit("CUDA preflight produced non-finite values")
        result.update(
            cuda=torch.version.cuda,
            gpu_count=torch.cuda.device_count(),
            gpu=torch.cuda.get_device_name(0),
            bf16_matmul="ok",
            flash_attention="ok",
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
