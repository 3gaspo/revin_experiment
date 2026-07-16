"""CSV loading, train/valid/test splits, and window sampling."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class TimeSeriesData:
    values: torch.Tensor  # users x 1 x dates

    def select(self, users, dates) -> "TimeSeriesData":
        return TimeSeriesData(self.values[users][:, :, dates])


def load_dataset(root: str, name: str, drop_users: list[int]) -> TimeSeriesData:
    frame = pd.read_csv(Path(root) / name / f"{name}.csv", index_col=0)
    if drop_users:
        frame = frame.drop(columns=frame.columns[drop_users])
    values = torch.tensor(frame.to_numpy(dtype=np.float32).T).unsqueeze(1)
    return TimeSeriesData(values)


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
    data = load_dataset(cfg.root, cfg.name, list(cfg.drop_users))
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
    return loaders, stats, data.values.shape[1]
