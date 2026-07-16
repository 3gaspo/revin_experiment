"""Forecasting model wrapper and normalization layers."""

from pathlib import Path

import torch
from torch import nn

from models.dlinear import DLinear
from models.patchtst import PatchTST


def normal_stats(x: torch.Tensor):
    return x.mean(-1, keepdim=True).detach(), x.std(-1, keepdim=True, unbiased=False).detach()


class IdentityNorm(nn.Module):
    def forward(self, x):
        return x

    def inverse(self, x):
        return x


class StandardNorm(nn.Module):
    def __init__(self, mean, std, eps=1e-8):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean))
        self.register_buffer("std", torch.as_tensor(std))
        self.eps = eps

    def forward(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def inverse(self, x):
        return x * (self.std + self.eps) + self.mean


class RevIN(nn.Module):
    def __init__(self, dim: int, affine=True, center="mean", transform=None, eps=1e-8):
        super().__init__()
        self.affine, self.center, self.transform, self.eps = affine, center, transform, eps
        if affine:
            self.weight = nn.Parameter(torch.ones(1, dim, 1))
            self.bias = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        self.shift = (x[..., -1:] if self.center == "last" else x.mean(-1, keepdim=True)).detach()
        self.scale = x.std(-1, keepdim=True, unbiased=False).detach()
        x = (x - self.shift) / (self.scale + self.eps)
        if self.affine:
            x = x * self.weight + self.bias
        return torch.asinh(x) if self.transform == "arcsinh" else x

    def inverse(self, x):
        if self.transform == "arcsinh":
            x = torch.sinh(x)
        if self.affine:
            x = (x - self.bias) / (self.weight + self.eps)
        return x * (self.scale + self.eps) + self.shift


class ForecastModel(nn.Module):
    def __init__(self, backbone, normalization):
        super().__init__()
        self.backbone, self.normalization = backbone, normalization

    def forward(self, x):
        return self.normalization.inverse(self.backbone(self.normalization(x)))

    def save(self, path):
        path = Path(path)
        torch.save(self.state_dict(), path)
        return path


def build_model(cfg, norm_cfg, lags: int, horizon: int, dim: int, stats):
    model_class = {"dlinear": DLinear, "patchtst": PatchTST}[cfg.name]
    backbone = model_class(lags=lags, dim=dim, horizon=horizon, **dict(cfg.kwargs))
    if norm_cfg.name == "none":
        normalization = IdentityNorm()
    elif norm_cfg.name == "standard":
        normalization = StandardNorm(**stats)
    else:
        normalization = RevIN(dim, **dict(norm_cfg.kwargs))
    return ForecastModel(backbone, normalization)
