"""DLinear forecasting architecture used as the experiment backbone."""

from __future__ import annotations

import torch
import torch.nn as nn


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) // 2
        front = x[:, :, :1].repeat(1, 1, pad)
        end = x[:, :, -1:].repeat(1, 1, pad)
        return self.avg(torch.cat([front, x, end], dim=-1))


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return residual, moving_mean


class DLinear(nn.Module):
    """Official DLinear decomposition and linear forecasting heads."""

    def __init__(
        self,
        lags: int,
        dim: int,
        horizon: int,
        kernel_size: int = 25,
        individual: bool = False,
    ):
        super().__init__()
        self.lags = int(lags)
        self.dim = int(dim)
        self.horizon = int(horizon)
        self.decomposition = SeriesDecomposition(kernel_size)
        self.individual = individual
        if individual:
            self.linear_seasonal = nn.ModuleList(
                [nn.Linear(self.lags, self.horizon) for _ in range(self.dim)]
            )
            self.linear_trend = nn.ModuleList(
                [nn.Linear(self.lags, self.horizon) for _ in range(self.dim)]
            )
            for seasonal, trend in zip(self.linear_seasonal, self.linear_trend):
                seasonal.weight = nn.Parameter(torch.ones(self.horizon, self.lags) / self.lags)
                trend.weight = nn.Parameter(torch.ones(self.horizon, self.lags) / self.lags)
        else:
            self.linear_seasonal = nn.Linear(self.lags, self.horizon)
            self.linear_trend = nn.Linear(self.lags, self.horizon)
            self.linear_seasonal.weight = nn.Parameter(
                torch.ones(self.horizon, self.lags) / self.lags
            )
            self.linear_trend.weight = nn.Parameter(
                torch.ones(self.horizon, self.lags) / self.lags
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal_init, trend_init = self.decomposition(x)
        if not self.individual:
            return self.linear_seasonal(seasonal_init) + self.linear_trend(trend_init)
        seasonal = x.new_empty(x.shape[0], self.dim, self.horizon)
        trend = torch.empty_like(seasonal)
        for i in range(self.dim):
            seasonal[:, i] = self.linear_seasonal[i](seasonal_init[:, i])
            trend[:, i] = self.linear_trend[i](trend_init[:, i])
        return seasonal + trend
