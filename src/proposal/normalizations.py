"""Readable normalization methods constituting the scientific proposal."""

import torch
from torch import nn


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
    CENTERS = {"mean", "last", "median", "none"}
    SCALES = {"std", "mad", "none"}

    def __init__(
        self,
        dim: int,
        affine=True,
        center="mean",
        scale="std",
        transform=None,
        eps=1e-8,
    ):
        super().__init__()
        if center not in self.CENTERS:
            raise ValueError(f"center must be one of {sorted(self.CENTERS)} (got {center!r})")
        if scale not in self.SCALES:
            raise ValueError(f"scale must be one of {sorted(self.SCALES)} (got {scale!r})")
        if transform not in {None, "arcsinh"}:
            raise ValueError("transform must be None or 'arcsinh'")
        self.affine = bool(affine)
        self.center = center
        self.scale_mode = scale
        self.transform = transform
        self.eps = float(eps)
        if affine:
            self.weight = nn.Parameter(torch.ones(1, dim, 1))
            self.bias = nn.Parameter(torch.zeros(1, dim, 1))

    def _location(self, x):
        if self.center == "none":
            return torch.zeros_like(x[..., :1])
        if self.center == "last":
            return x[..., -1:]
        if self.center == "median":
            return x.median(-1, keepdim=True).values
        return x.mean(-1, keepdim=True)

    def _scale(self, x):
        if self.scale_mode == "none":
            return torch.ones_like(x[..., :1])
        if self.scale_mode == "mad":
            median = x.median(-1, keepdim=True).values
            return (x - median).abs().median(-1, keepdim=True).values
        return x.std(-1, keepdim=True, unbiased=False)

    def _denominator(self):
        return self.scale_value if self.scale_mode == "none" else self.scale_value + self.eps

    def forward(self, x):
        self.shift = self._location(x).detach()
        self.scale_value = self._scale(x).detach()
        x = (x - self.shift) / self._denominator()
        if self.affine:
            x = x * self.weight + self.bias
        return torch.asinh(x) if self.transform == "arcsinh" else x

    def _inverse_standardized(self, x):
        if self.transform == "arcsinh":
            x = torch.sinh(x)
        if self.affine:
            x = (x - self.bias) / (self.weight + self.eps)
        return x

    def _restore(self, x):
        return x * self._denominator() + self.shift

    def inverse(self, x):
        return self._restore(self._inverse_standardized(x))


class MIN(RevIN):
    """Global modulated instance normalization with untied output affine."""

    def __init__(self, dim: int, affine=True, center="mean", scale="std", transform=None, eps=1e-8):
        if not affine:
            raise ValueError("MIN requires its input affine parameters")
        super().__init__(dim, affine=True, center=center, scale=scale, transform=transform, eps=eps)
        self.output_scale = nn.Parameter(torch.ones(1, dim, 1))
        self.output_shift = nn.Parameter(torch.zeros(1, dim, 1))

    def inverse(self, x):
        x = self._inverse_standardized(x)
        x = x * self.output_scale + self.output_shift
        return self._restore(x)
