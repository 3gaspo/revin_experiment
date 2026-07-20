"""CSV loading, train/valid/test splits, and window sampling."""

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TimeSeriesData:
    values: torch.Tensor  # users x 1 x dates

    def select(self, users, dates) -> "TimeSeriesData":
        return TimeSeriesData(self.values[users][:, :, dates])


def _drop_user_list(value: Any) -> list[int]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        values = [
            part.strip()
            for part in value.replace(";", ",").split(",")
            if part.strip()
        ]
    elif isinstance(value, Iterable):
        values = list(value)
    else:
        values = [value]
    return [int(item) for item in values]


def _merge_drop_users(*values: Any) -> list[int]:
    return list(dict.fromkeys(item for value in values for item in _drop_user_list(value)))


def _column_names(value: Any) -> list[str] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, Iterable):
        values = list(value)
    else:
        values = [value]
    return [str(item) for item in values]


def _dataset_config_path(
    root: str | Path,
    name: str,
    config_path: str | Path | None,
) -> tuple[Path, bool]:
    if config_path is not None and str(config_path) != "":
        path = Path(config_path).expanduser()
        return (path / "config.json" if path.is_dir() else path), True
    return Path(root).expanduser() / name / "config.json", False


def _load_dataset_config(
    root: str | Path,
    name: str,
    config_path: str | Path | None,
) -> tuple[dict[str, Any], Path | None]:
    path, explicit = _dataset_config_path(root, name, config_path)
    if not path.is_file():
        if explicit:
            raise FileNotFoundError(f"dataset config does not exist: {path}")
        return {}, None
    if path.suffix.lower() != ".json":
        raise ValueError(f"dataset config must be JSON, got {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"dataset config must contain a JSON object: {path}")
    options = {
        "drop_users": _drop_user_list(raw.get("drop_users")),
        "target_cols": _column_names(raw.get("target_cols")),
    }
    scoped = raw.get("revin")
    if scoped is not None:
        if not isinstance(scoped, Mapping):
            raise ValueError("dataset config field 'revin' must be an object")
        options["drop_users"] = _merge_drop_users(
            options.get("drop_users"), scoped.get("drop_users")
        )
        if scoped.get("target_cols") is not None:
            options["target_cols"] = _column_names(scoped.get("target_cols"))
    LOGGER.info("loaded dataset config path=%s keys=%s", path, sorted(options))
    return options, path


def load_dataset(
    root: str,
    name: str,
    drop_users: Any = None,
    config_path: str | Path | None = None,
    target_cols: Any = None,
) -> tuple[TimeSeriesData, dict[str, Any]]:
    csv_path = Path(root) / name / f"{name}.csv"
    frame = pd.read_csv(csv_path, index_col=0)
    config, loaded_config_path = _load_dataset_config(root, name, config_path)
    configured_drops = _drop_user_list(config.get("drop_users"))
    run_drops = _drop_user_list(drop_users)
    applied_drops = _merge_drop_users(configured_drops, run_drops)
    configured_targets = _column_names(config.get("target_cols"))
    run_targets = _column_names(target_cols)
    applied_targets = run_targets if run_targets is not None else configured_targets
    invalid = [
        index for index in applied_drops if index < 0 or index >= len(frame.columns)
    ]
    if invalid:
        raise ValueError(f"drop_users contains invalid source positions {invalid}")
    dropped_columns = [str(frame.columns[index]) for index in applied_drops]
    if applied_drops:
        frame = frame.drop(columns=frame.columns[applied_drops])
    if applied_targets is not None:
        missing = [column for column in applied_targets if column not in frame.columns]
        if missing:
            raise ValueError(f"target_cols contains missing or dropped columns {missing}")
        frame = frame.loc[:, applied_targets]
    if frame.shape[1] == 0:
        raise ValueError("dataset has no users after applying drop_users")
    values = torch.tensor(frame.to_numpy(dtype=np.float32).T).unsqueeze(1)
    metadata = {
        "csv_path": str(csv_path),
        "config_path": None if loaded_config_path is None else str(loaded_config_path),
        "config_keys": sorted(config),
        "drop_users_from_config": configured_drops,
        "drop_users_from_run": run_drops,
        "drop_users_applied": applied_drops,
        "dropped_columns": dropped_columns,
        "target_cols_from_config": configured_targets,
        "target_cols_from_run": run_targets,
        "target_cols_applied": applied_targets,
        "retained_users": int(frame.shape[1]),
    }
    LOGGER.info(
        "loaded dataset name=%s users=%s dropped_users=%s config=%s",
        name,
        frame.shape[1],
        applied_drops,
        loaded_config_path,
    )
    return TimeSeriesData(values), metadata


def split_dataset(data: TimeSeriesData, date_splits, indiv_split: float, seed: int):
    n_users, _, n_dates = data.values.shape
    train_end = int(date_splits[0] * n_dates)
    valid_end = int((date_splits[0] + date_splits[1]) * n_dates)
    dates = [range(train_end), range(train_end, valid_end), range(valid_end, n_dates)]

    if indiv_split == 1:
        return {
            name: data.select(range(n_users), block)
            for name, block in zip(("train", "valid", "test"), dates)
        }

    users = np.random.default_rng(seed).permutation(n_users)
    cut = int(indiv_split * n_users)
    groups = [users[:cut], users[cut:]]
    names = [("train", "valid1", "test1"), ("train2", "valid2", "test2")]
    return {
        name: data.select(group, block)
        for group, group_names in zip(groups, names)
        for name, block in zip(group_names, dates)
    }


def window(data: TimeSeriesData, user: int, start: int, lags: int, horizon: int):
    values = data.values[user, :, start : start + lags + horizon]
    return values[:, :lags], values[:, lags:]


class RandomWindows(Dataset):
    def __init__(self, data: TimeSeriesData, lags: int, horizon: int):
        self.data, self.lags, self.horizon = data, lags, horizon
        self.starts = data.values.shape[-1] - lags - horizon + 1

    def __len__(self):
        return len(self.data.values)

    def __getitem__(self, _):
        user = np.random.randint(len(self.data.values))
        start = np.random.randint(self.starts)
        return window(self.data, user, start, self.lags, self.horizon)


class AllWindows(Dataset):
    def __init__(self, data: TimeSeriesData, lags: int, horizon: int, stride: int):
        self.data, self.lags, self.horizon = data, lags, horizon
        starts = range(0, data.values.shape[-1] - lags - horizon + 1, stride)
        self.indices = [(user, start) for start in starts for user in range(len(data.values))]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        user, start = self.indices[index]
        return window(self.data, user, start, self.lags, self.horizon)


def build_loaders(cfg, lags: int, horizon: int, batch_size: int, seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    data, metadata = load_dataset(
        cfg.root,
        cfg.name,
        cfg.get("drop_users"),
        cfg.get("config_path"),
        cfg.get("target_cols"),
    )
    splits = split_dataset(data, cfg.date_splits, cfg.indiv_split, seed)
    loaders = {
        name: DataLoader(
            RandomWindows(split, lags, horizon)
            if name == "train"
            else AllWindows(split, lags, horizon, cfg.eval_stride),
            batch_size=batch_size,
        )
        for name, split in splits.items()
    }
    train = splits["train"].values
    stats = {"mean": train.mean(), "std": train.std(unbiased=False)}
    return loaders, stats, data.values.shape[1], metadata
