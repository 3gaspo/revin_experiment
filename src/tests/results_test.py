"""Dependency-free checks for RevIN result aggregation."""

import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.results import generate_average_summary, generate_results_table


def write_result(root, dataset, method, seed, mean):
    output = root / dataset / "168_24" / method / f"seed_{seed}"
    output.mkdir(parents=True)
    payload = {
        "test1": {
            "mse": {
                "mean": mean,
                "std": 2.0,
                "variance": 4.0,
                "count": 10,
            }
        }
    }
    (output / "results.json").write_text(json.dumps(payload), encoding="utf-8")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for dataset, standard, instance in (
            ("first", (10.0, 12.0), (8.0, 10.0)),
            ("second", (20.0, 22.0), (22.0, 24.0)),
        ):
            for seed, mean in enumerate(standard, 1):
                write_result(root, dataset, "patchtst_standard_mse", seed, mean)
            for seed, mean in enumerate(instance, 1):
                write_result(root, dataset, "patchtst_instance_nmse", seed, mean)

        table = generate_results_table(
            root,
            metric="mse",
            split="test1",
            settings=["168:24"],
            show_std=True,
        )
        assert "$\\pm$" in table.read_text(encoding="utf-8")

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
        assert summary["oracle"]["selection_counts"] == {
            "patchtst_instance_nmse": 1,
            "patchtst_standard_mse": 1,
        }
        assert summary_tex.exists()


if __name__ == "__main__":
    main()
