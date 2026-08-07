"""One-shot migration of reconstructable pre-schema RevIN outputs."""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from experiment_runs import _atomic_json, allocate_run, identity_path, load_manifest, mark_status


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = PROJECT_ROOT / "outputs"
LEGACY_ROOTS = (OUTPUTS / "revin_experiment", OUTPUTS / "revin_experiment_test")
ARCHIVE = OUTPUTS / "archive" / "legacy_pre_schema_v1_2026-08-07"

CORE = {"none_mse", "standard_mse", "instance_mse", "instance_nmse", "revin_mse", "revin_nmse"}
NMSE = {"none_nmse", "standard_nmse"}
MIN = {"min_mse", "min_nmse"}


def marker(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for item in path.read_text(encoding="utf-8").strip().split("|"):
        if "=" in item:
            key, value = item.split("=", 1)
            fields[key] = value
    required = {
        "dataset", "lags", "horizon", "model", "method", "epochs", "steps",
        "batch_size", "learning_rate", "eval_stride", "valid_eval_freq",
        "logging_eval_freq", "dataset_config_sha256", "seed",
    }
    missing = required - fields.keys()
    if missing:
        raise ValueError(f"cannot reconstruct {path}: missing marker fields {sorted(missing)}")
    return fields


def scalar(value: str):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def family(method: str, *, smoke: bool) -> str:
    if smoke or method in CORE:
        return "core"
    if method in NMSE:
        return "nmse"
    if method in MIN:
        return "min"
    return "exotic"


def discover():
    grouped = defaultdict(list)
    for legacy_root in LEGACY_ROOTS:
        if not legacy_root.is_dir():
            continue
        for marker_path in sorted(legacy_root.glob("*/*/*/seed_*/run.complete")):
            values = marker(marker_path)
            scientific = tuple(sorted((key, value) for key, value in values.items() if key != "seed"))
            grouped[(legacy_root, scientific)].append((marker_path.parent, values))
    return grouped


def migrate_group(legacy_root: Path, seeds: list[tuple[Path, dict[str, str]]]) -> Path:
    first = seeds[0][1]
    method = first["method"]
    backbone = first["model"]
    suffix = method.removeprefix(f"{backbone}_")
    normalization, loss = suffix.rsplit("_", 1)
    workflow = family(suffix, smoke=legacy_root.name.endswith("_test"))
    identity_root = identity_path(
        OUTPUTS / workflow,
        first["dataset"],
        int(first["lags"]),
        int(first["horizon"]),
        backbone,
        ["normalization", "loss"],
        {"normalization": normalization, "loss": loss},
    )
    pipeline = {
        "training.epochs": scalar(first["epochs"]),
        "training.steps": scalar(first["steps"]),
        "training.batch_size": scalar(first["batch_size"]),
        "training.learning_rate": scalar(first["learning_rate"]),
        "evaluation.stride": scalar(first["eval_stride"]),
        "training.valid_eval_freq": scalar(first["valid_eval_freq"]),
        "training.logging_eval_freq": scalar(first["logging_eval_freq"]),
    }
    seed_numbers = sorted(int(values["seed"]) for _, values in seeds)
    allocation = allocate_run(
        identity_root,
        project="revin_experiment",
        workflow=workflow,
        dataset=first["dataset"],
        lookback=int(first["lags"]),
        horizon=int(first["horizon"]),
        backbone=backbone,
        model_config_order=["normalization", "loss"],
        model_config={"normalization": normalization, "loss": loss},
        pipeline_config=pipeline,
        runtime_config={"migration": "local"},
        seeds=seed_numbers,
        purpose="smoke" if legacy_root.name.endswith("_test") else "publication",
        mode="test" if legacy_root.name.endswith("_test") else None,
        display_name=method,
        row_config=["normalization"],
        column_config=["loss"],
        inputs={"dataset_config": {"sha256": first["dataset_config_sha256"]}},
        code_signature="legacy-revin-run-marker-v2",
        policy="new",
        skip_completed=False,
        launch_id=f"migration-{legacy_root.name}-{method}",
    )
    required: list[str] = []
    original_starts: list[float] = []
    original_finishes: list[float] = []
    for source, values in seeds:
        seed = int(values["seed"])
        destination = allocation.run_dir / f"seed_{seed}"
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copytree(source, destination)
        (destination / "run.complete").unlink()
        artifacts = [f"seed_{seed}/{name}" for name in ("results.json", "config.yaml", "dataset_config.json")]
        for artifact in artifacts:
            path = allocation.run_dir / artifact
            if not path.is_file() or path.stat().st_size <= 0:
                raise FileNotFoundError(path)
        mark_status(allocation.run_dir, "completed", required_artifacts=artifacts, seed=seed)
        original_starts.append((source / "config.yaml").stat().st_mtime)
        original_finishes.append((source / "results.json").stat().st_mtime)
        required.extend(artifacts)
    mark_status(allocation.run_dir, "completed", required_artifacts=required)
    manifest = load_manifest(allocation.run_dir)
    manifest["migration"] = {
        "source": str(legacy_root),
        "migrated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence": "complete v2 run marker plus config.yaml, dataset_config.json, and results.json",
    }
    manifest["launch"]["launched_at_utc"] = datetime.fromtimestamp(min(original_starts), timezone.utc).isoformat()
    manifest["launch"]["started_at_utc"] = manifest["launch"]["launched_at_utc"]
    manifest["launch"]["finished_at_utc"] = datetime.fromtimestamp(max(original_finishes), timezone.utc).isoformat()
    _atomic_json(allocation.run_dir / "manifest.json", manifest)
    return allocation.run_dir


def main() -> None:
    groups = discover()
    if not groups:
        raise RuntimeError("no legacy RevIN seed markers found")
    migrated = [migrate_group(root, seeds) for (root, _), seeds in groups.items()]
    for run_dir in migrated:
        manifest = load_manifest(run_dir)
        if manifest["status"] != "completed":
            raise RuntimeError(f"migration did not complete {run_dir}")
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for legacy_root in LEGACY_ROOTS:
        if legacy_root.exists():
            destination = ARCHIVE / legacy_root.name
            if destination.exists():
                raise FileExistsError(destination)
            shutil.move(str(legacy_root), destination)
    print(json.dumps({"migrated_runs": [str(path) for path in migrated], "archive": str(ARCHIVE)}, indent=2))


if __name__ == "__main__":
    main()
