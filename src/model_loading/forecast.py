"""Forecasting-model construction and normalization composition."""

from pathlib import Path

import torch
from torch import nn

from external_models.dlinear import DLinear
from external_models.patchtst import PatchTST
from proposal.normalizations import IdentityNorm, MIN, RevIN, StandardNorm


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
    elif norm_cfg.name == "min":
        normalization = MIN(dim, **dict(norm_cfg.kwargs))
    else:
        normalization = RevIN(dim, **dict(norm_cfg.kwargs))
    return ForecastModel(backbone, normalization)
