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

RESAMPLE_REDUCERS = {"sum", "mean", "last", "first"}


@dataclass(frozen=True)
class TimeSeriesData:
    values: torch.Tensor  # users x 1 x dates
    target_start: int = 0
    target_end: int | None = None

    @property
    def dates(self) -> int:
        return int(self.values.shape[-1])

    @property
    def target_bounds(self) -> tuple[int, int]:
        end = self.dates if self.target_end is None else int(self.target_end)
        return int(self.target_start), end

    @property
    def target_values(self) -> torch.Tensor:
        start, end = self.target_bounds
        return self.values[..., start:end]

    def select(self, users, target_start: int, target_end: int) -> "TimeSeriesData":
        users = [int(user) for user in users]
        values = (
            self.values
            if users == list(range(len(self.values)))
            else self.values[users]
        )
        return TimeSeriesData(
            values,
            target_start=int(target_start),
            target_end=int(target_end),
        )


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
        "date_col": raw.get("date_col"),
        "aggr": raw.get("aggr"),
        "aggr_period": raw.get("aggr_period"),
        "missing_values": raw.get("missing_values"),
    }
    scoped = raw.get("revin")
    if scoped is not None:
        if not isinstance(scoped, Mapping):
            raise ValueError("dataset config field 'revin' must be an object")
        if scoped.get("drop_users") is not None:
            options["drop_users"] = _drop_user_list(scoped.get("drop_users"))
        if scoped.get("target_cols") is not None:
            options["target_cols"] = _column_names(scoped.get("target_cols"))
        for key in ("date_col", "aggr", "aggr_period", "missing_values"):
            if scoped.get(key) is not None:
                options[key] = scoped[key]
    LOGGER.info("loaded dataset config path=%s keys=%s", path, sorted(options))
    return options, path


def load_dataset(
    root: str,
    name: str,
    drop_users: Any = None,
    config_path: str | Path | None = None,
    target_cols: Any = None,
    date_col: str | None = None,
    aggr: str | None = None,
    aggr_period: str | None = None,
    missing_values: str | None = None,
) -> tuple[TimeSeriesData, dict[str, Any]]:
    csv_path = Path(root) / name / f"{name}.csv"
    config, loaded_config_path = _load_dataset_config(root, name, config_path)
    applied_date_col = date_col if date_col is not None else config.get("date_col")
    applied_aggr = aggr if aggr is not None else config.get("aggr")
    applied_aggr_period = (
        aggr_period
        if aggr_period is not None
        else config.get("aggr_period") or "h"
    )
    applied_missing_values = str(
        missing_values
        if missing_values is not None
        else config.get("missing_values") or "zero"
    ).lower()
    if applied_missing_values not in {"zero", "error"}:
        raise ValueError("missing_values must be 'zero' or 'error'")
    if applied_date_col:
        frame = pd.read_csv(csv_path, parse_dates=[applied_date_col])
        frame = frame.set_index(applied_date_col)
    else:
        frame = pd.read_csv(csv_path, index_col=0)
        frame.index = pd.to_datetime(frame.index)
    source_infinite_count = int(np.isinf(frame.to_numpy(dtype=float)).sum())
    if source_infinite_count:
        raise ValueError(f"dataset contains {source_infinite_count} infinite values")
    if applied_aggr is not None and str(applied_aggr).lower() not in {"", "none"}:
        reducer = str(applied_aggr).lower()
        if reducer == "asfreq":
            frame = frame.asfreq(str(applied_aggr_period))
        elif reducer in RESAMPLE_REDUCERS:
            frame = getattr(frame.resample(str(applied_aggr_period)), reducer)()
        else:
            raise ValueError(f"unknown aggregation {applied_aggr!r}")
    missing_count = int(frame.isna().sum().sum())
    if missing_count and applied_missing_values == "error":
        raise ValueError(f"dataset contains {missing_count} missing values")
    if missing_count:
        frame = frame.fillna(0.0)
    infinite_count = int(np.isinf(frame.to_numpy(dtype=float)).sum())
    if infinite_count:
        raise ValueError(f"dataset contains {infinite_count} infinite values")
    configured_drops = _drop_user_list(config.get("drop_users"))
    run_drops = _drop_user_list(drop_users)
    applied_drops = run_drops if drop_users is not None else configured_drops
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
        "window_anchor": "query_t",
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
        "date_col_applied": applied_date_col,
        "aggr_applied": applied_aggr,
        "aggr_period_applied": applied_aggr_period,
        "missing_values_applied": applied_missing_values,
        "missing_values_replaced": missing_count,
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
    target_bounds = [(0, train_end), (train_end, valid_end), (valid_end, n_dates)]

    if indiv_split == 1:
        selected = data.select(range(n_users), 0, n_dates)
        return {
            name: TimeSeriesData(selected.values, *bounds)
            for name, bounds in zip(("train", "valid", "test"), target_bounds)
        }

    users = np.random.default_rng(seed).permutation(n_users)
    cut = int(indiv_split * n_users)
    groups = [users[:cut], users[cut:]]
    names = [("train", "valid1", "test1"), ("train2", "valid2", "test2")]
    splits = {}
    for group, group_names in zip(groups, names):
        selected = data.select(group, 0, n_dates)
        for name, bounds in zip(group_names, target_bounds):
            splits[name] = TimeSeriesData(selected.values, *bounds)
    return splits


def query_dates(
    data: TimeSeriesData,
    lags: int,
    horizon: int,
    stride: int = 1,
) -> range:
    """Return cutoff dates whose complete targets belong to this split."""
    target_start, target_end = data.target_bounds
    first = max(int(lags) - 1, target_start - 1)
    last = min(data.dates - int(horizon) - 1, target_end - int(horizon) - 1)
    if last < first:
        return range(0)
    return range(first, last + 1, int(stride))


def window(data: TimeSeriesData, user: int, query_t: int, lags: int, horizon: int):
    """Return ``X=(t-L,t]`` and ``Y=(t,t+H]`` for cutoff ``query_t``."""
    start = int(query_t) - int(lags) + 1
    stop = int(query_t) + int(horizon) + 1
    values = data.values[user, :, start:stop]
    return values[:, :lags], values[:, lags:]


class RandomWindows(Dataset):
    def __init__(self, data: TimeSeriesData, lags: int, horizon: int):
        self.data, self.lags, self.horizon = data, lags, horizon
        self.query_dates = tuple(query_dates(data, lags, horizon))
        if not self.query_dates:
            raise ValueError("split has no query date with a complete target")

    def __len__(self):
        return len(self.data.values)

    def __getitem__(self, _):
        user = np.random.randint(len(self.data.values))
        query_t = int(np.random.choice(self.query_dates))
        return window(self.data, user, query_t, self.lags, self.horizon)


class AllWindows(Dataset):
    def __init__(self, data: TimeSeriesData, lags: int, horizon: int, stride: int):
        self.data, self.lags, self.horizon = data, lags, horizon
        dates = query_dates(data, lags, horizon, stride)
        self.indices = [
            (user, query_t)
            for query_t in dates
            for user in range(len(data.values))
        ]
        if not self.indices:
            raise ValueError("split has no query date with a complete target")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        user, query_t = self.indices[index]
        return window(self.data, user, query_t, self.lags, self.horizon)


def build_loaders(cfg, lags: int, horizon: int, batch_size: int, seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    data, metadata = load_dataset(
        root=cfg.root,
        name=cfg.name,
        drop_users=cfg.get("drop_users"),
        config_path=cfg.get("config_path"),
        target_cols=cfg.get("target_cols"),
        date_col=cfg.get("date_col"),
        aggr=cfg.get("aggr"),
        aggr_period=cfg.get("aggr_period"),
        missing_values=cfg.get("missing_values"),
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
    train = splits["train"].target_values
    stats = {"mean": train.mean(), "std": train.std(unbiased=False)}
    return loaders, stats, data.values.shape[1], metadata
