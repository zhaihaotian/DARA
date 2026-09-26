"""Assemble the DARA math runtime: public verl v0.7.0 plus the 12-file overlay in runtime/overlay.

The overlay contains the modified files, not the whole verl distribution or
its GPU dependencies. Assembly and syntax checks do not establish GPU readiness.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "runtime/overlay"
MANIFEST = ROOT / "runtime/overlay.sha256"


def prepare(base, output):
    base, output = base.resolve(), output.resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}; choose a new directory")
    if base == output or base in output.parents:
        raise ValueError("Output must be outside the source framework")
    version_file = base / "verl/version/version"
    if not version_file.is_file() or not version_file.read_text().strip().startswith("0.7"):
        raise ValueError("Supply a complete verl 0.7 source tree, not the overlay alone")
    for rel in ("verl/trainer/main_ppo.py", "verl/trainer/ppo/core_algos.py"):
        if not (base / rel).is_file():
            raise ValueError(f"Incomplete base framework: missing {rel}")
    bundle_files = sorted(p.relative_to(BUNDLE) for p in BUNDLE.rglob("*") if p.is_file())
    bundle_files = [p for p in bundle_files if p.suffix in (".py", ".yaml")]
    if len(bundle_files) != 12:
        raise ValueError("Expected the 12 published runtime source files")
    manifest_path = MANIFEST
    manifest = {}
    for line in manifest_path.read_text().splitlines():
        if line and not line.startswith("#"):
            digest, relative_path = line.split(maxsplit=1)
            manifest[Path(relative_path)] = digest
    if set(manifest) != set(bundle_files):
        raise ValueError("Runtime source manifest does not match the 12 overlay files")
    for rel in bundle_files:
        digest = hashlib.sha256((BUNDLE / rel).read_bytes()).hexdigest()
        if digest != manifest[rel]:
            raise ValueError(f"Runtime source hash mismatch: {rel}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rdgdpo-prepare-", dir=output.parent) as tmp:
        stage = Path(tmp) / "framework"
        shutil.copytree(base, stage, ignore=shutil.ignore_patterns(
            ".git", "__pycache__", ".venv", "wandb", "checkpoints", "*.pyc"))
        for rel in bundle_files:
            (stage / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(BUNDLE / rel, stage / rel)
        rollout_registry = stage / "verl/workers/rollout/base.py"
        registry_source = rollout_registry.read_text()
        async_entry = '    ("vllm", "async"): "verl.workers.rollout.vllm_rollout.vLLMAsyncRollout",\n'
        sync_entry = (
            '    ("vllm", "sync"): '
            '"verl.workers.rollout.vllm_rollout.vllm_rollout_spmd.vLLMRollout",\n'
        )
        if sync_entry not in registry_source:
            if async_entry not in registry_source:
                raise ValueError("Compatible verl 0.7 rollout registry entry was not found")
            rollout_registry.write_text(registry_source.replace(async_entry, sync_entry + async_entry, 1))
        shutil.copytree(ROOT / "rdgdpo", stage / "rdgdpo", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        dataset_source = stage / "verl/utils/dataset/rl_dataset.py"
        dataset_text = dataset_source.read_text()
        raw_prompt_marker = "apply_chat_template has been moved to AgentLoop"
        dataset_adapter = (
            "\n\nfrom rdgdpo.sync_dataset import sync_text_getitem\n\n"
            "RLHFDataset.__getitem__ = sync_text_getitem\n"
        )
        if raw_prompt_marker not in dataset_text:
            raise ValueError("Compatible verl 0.7 raw-prompt dataset implementation was not found")
        if dataset_adapter not in dataset_text:
            dataset_source.write_text(dataset_text + dataset_adapter)
        rollout_helper = stage / "verl/trainer/ppo/rollout_corr_helper.py"
        rollout_helper_text = rollout_helper.read_text()
        rollout_helper_marker = "def apply_bypass_mode("
        rollout_helper_adapter = (
            "\n\nfrom rdgdpo.rollout_corr_compat import maybe_apply_rollout_correction\n"
        )
        if rollout_helper_marker not in rollout_helper_text:
            raise ValueError("Compatible verl 0.7 rollout correction helper was not found")
        if ("def maybe_apply_rollout_correction(" not in rollout_helper_text
                and rollout_helper_adapter not in rollout_helper_text):
            rollout_helper.write_text(rollout_helper_text + rollout_helper_adapter)
        compat_paths = [rollout_registry, dataset_source, rollout_helper]
        for path in ([stage / rel for rel in bundle_files if rel.suffix == ".py"]
                     + list((stage / "rdgdpo").glob("*.py")) + compat_paths):
            ast.parse(path.read_text(), filename=str(path))
        (stage / "rdgdpo_runtime.json").write_text(json.dumps({
            "profile": "dara-math", "source": str(base), "recenter": False,
            "verification": "source_assembly_only", "gpu_validated": False,
            "methods": ["grpo", "gdpo", "dvao", "gd2po_hard", "rdgdpo"],
        }, indent=2)+"\n")
        shutil.move(str(stage), str(output))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-framework", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(prepare(args.base_framework, args.output))
    except (ValueError, OSError) as error:
        parser.exit(2, str(error)+"\n")


if __name__ == "__main__":
    main()
