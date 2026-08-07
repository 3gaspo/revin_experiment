"""Dependency-free checks for RevIN result aggregation."""

import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_runs import allocate_run, mark_status
from utils.results import generate_average_summary, generate_results_table


def write_run(root, dataset, method, means, valid_means):
    _, normalization, loss = method.split("_", 2)
    identity = root / dataset / "168_24" / "patchtst" / normalization / loss
    seeds = list(range(1, len(means) + 1))
    allocation = allocate_run(
        identity,
        project="revin_experiment",
        workflow="core",
        dataset=dataset,
        lookback=168,
        horizon=24,
        backbone="patchtst",
        model_config_order=["normalization", "loss"],
        model_config={"normalization": normalization, "loss": loss},
        pipeline_config={},
        seeds=seeds,
        display_name=method,
    )
    artifacts = []
    for seed, (mean, valid_mean) in enumerate(zip(means, valid_means), 1):
        relative = f"seed_{seed}/results.json"
        output = allocation.run_dir / relative
        output.parent.mkdir(parents=True)
        payload = {
            "valid1": {"mse": {"mean": valid_mean}},
            "test1": {
                "mse": {
                    "mean": mean,
                    "std": 2.0,
                    "variance": 4.0,
                    "count": 10,
                }
            }
        }
        output.write_text(json.dumps(payload), encoding="utf-8")
        mark_status(allocation.run_dir, "completed", seed=seed, required_artifacts=[relative])
        artifacts.append(relative)
    mark_status(allocation.run_dir, "completed", required_artifacts=artifacts)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for dataset, standard, instance, standard_valid, instance_valid in (
            (
                "first",
                (10.0, 12.0),
                (8.0, 10.0),
                (10.0, 10.0),
                (8.0, 8.0),
            ),
            (
                "second",
                (20.0, 22.0),
                (22.0, 24.0),
                (22.0, 22.0),
                (21.0, 21.0),
            ),
        ):
            write_run(root, dataset, "patchtst_standard_mse", standard, standard_valid)
            write_run(root, dataset, "patchtst_instance_nmse", instance, instance_valid)

        table = generate_results_table(
            root,
            metric="mse",
            split="test1",
            settings=["168:24"],
            show_std=True,
            selection_methods=[
                "patchtst_standard_mse",
                "patchtst_instance_nmse",
            ],
        )
        table_text = table.read_text(encoding="utf-8")
        assert "$\\pm$" in table_text
        assert "validation-selected & test-oracle" in table_text
        assert "llcrrr|r" in table_text

        summary_json, summary_tex = generate_average_summary(
            root,
            root / "summary.json",
            datasets=["first", "second"],
            settings=["168:24"],
            methods=["patchtst_standard_mse", "patchtst_instance_nmse"],
            oracle_methods=["patchtst_standard_mse", "patchtst_instance_nmse"],
            baseline_method="patchtst_standard_mse",
            expected_seeds=[1, 2],
            strict=True,
        )
        summary = json.loads(summary_json.read_text(encoding="utf-8"))
        assert math.isclose(
            summary["methods"]["patchtst_standard_mse"]["macro_mean"], 16.0
        )
        assert math.isclose(
            summary["methods"]["patchtst_instance_nmse"]["macro_mean"], 16.0
        )
        assert math.isclose(summary["oracle"]["macro_mean"], 15.0)
        assert math.isclose(summary["oracle"]["seed_variance"], 2.0)
        assert math.isclose(summary["validation_selected"]["macro_mean"], 16.0)
        assert summary["validation_selected"]["selection_counts"] == {
            "patchtst_instance_nmse": 2,
        }
        assert summary["oracle"]["selection_counts"] == {
            "patchtst_instance_nmse": 1,
            "patchtst_standard_mse": 1,
        }
        assert summary_tex.exists()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_run(root, "electricity", "patchtst_standard_mse", [2.0], [2.5])
        table = generate_results_table(
            root,
            metric="mse",
            split="test1",
            settings=["168:24"],
            methods=["patchtst_standard_mse"],
            show_std=True,
        )
        table_text = table.read_text(encoding="utf-8")
        assert "$\\pm$" not in table_text
        assert "When at least two seeds are available" in table_text
        summary_json, summary_tex = generate_average_summary(
            root,
            root / "summary.json",
            datasets=["electricity"],
            settings=["168:24"],
            methods=["patchtst_standard_mse"],
            oracle_methods=["patchtst_standard_mse"],
            baseline_method="patchtst_standard_mse",
            expected_seeds=[1],
            strict=True,
        )
        summary = json.loads(summary_json.read_text(encoding="utf-8"))
        assert summary["methods"]["patchtst_standard_mse"]["seed_std"] is None
        assert "--" in summary_tex.read_text(encoding="utf-8")


if __name__ == "__main__":
    main()
