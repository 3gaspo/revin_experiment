"""Focused CSV missing-value policy checks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data.dataset import load_dataset


class DatasetLoadingTest(unittest.TestCase):
    def test_missing_values_default_to_zero_and_infinity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "tiny"
            dataset.mkdir()
            path = dataset / "tiny.csv"
            pd.DataFrame({"a": [1.0, np.nan]}).to_csv(path)

            filled, metadata = load_dataset(root, "tiny")
            self.assertEqual(filled.values[0, 0, 1].item(), 0.0)
            self.assertEqual(metadata["missing_values_replaced"], 1)
            with self.assertRaisesRegex(ValueError, "missing values"):
                load_dataset(root, "tiny", missing_values="error")

            pd.DataFrame({"a": [1.0, np.inf]}).to_csv(path)
            with self.assertRaisesRegex(ValueError, "infinite values"):
                load_dataset(root, "tiny")


if __name__ == "__main__":
    unittest.main()
