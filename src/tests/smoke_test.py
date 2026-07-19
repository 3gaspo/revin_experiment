"""One small CPU run through training, evaluation, and table generation."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.experiment import run_experiment
from utils.results import generate_results_table


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dataset = root / "datasets" / "tiny"
        dataset.mkdir(parents=True)
        t = np.arange(80)
        pd.DataFrame(
            {
                "a": np.sin(t / 5),
                "b": np.cos(t / 7),
                "c": np.sin(t / 9) + 1,
                "d": np.cos(t / 11) + 2,
                "e": np.sin(t / 13) + 3,
            }
        ).to_csv(dataset / "tiny.csv")
        (dataset / "config.json").write_text(
            json.dumps({"drop_users": [0], "revin": {"drop_users": [1]}}),
            encoding="utf-8",
        )

        output = root / "outputs"
        cfg = OmegaConf.create(
            {
                "data": {
                    "root": str(root / "datasets"),
                    "name": "tiny",
                    "config_path": None,
                    "drop_users": [2],
                    "date_splits": [0.5, 0.25, 0.25],
                    "indiv_split": 0.5,
                    "eval_stride": 4,
                },
                "task": {"lags": 12, "horizon": 4},
                "model": {"name": "dlinear", "kwargs": {}},
                "normalization": {"name": "revin", "kwargs": {"affine": False}},
                "training": {
                    "batch_size": 4,
                    "epochs": 1,
                    "lr": 1e-5,
                    "loss": "nmse",
                    "device": "cpu",
                },
                "seed": 1,
                "output": {
                    "dir": str(output / "tiny" / "12_4"),
                    "name": "dlinear_instance/seed_1",
                },
            }
        )
        run_experiment(cfg)
        run = output / "tiny" / "12_4" / "dlinear_instance" / "seed_1"
        assert (run / "model.pt").exists()
        assert (run / "results.json").exists()
        metadata = json.loads((run / "dataset_config.json").read_text(encoding="utf-8"))
        assert metadata["drop_users_applied"] == [0, 1, 2]
        assert metadata["retained_users"] == 2
        table = generate_results_table(
            output,
            metric="mse",
            split="test1",
            methods=["dlinear_instance"],
            show_std=True,
        )
        assert table.exists()


if __name__ == "__main__":
    main()
