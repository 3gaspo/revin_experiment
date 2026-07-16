"""Losses and the training/evaluation loop."""

import random

import numpy as np
import torch
from torch import nn

from utils.models import normal_stats


class ForecastLoss(nn.Module):
    def __init__(self, name="nmse", reduction="mean", eps=1e-8):
        super().__init__()
        self.name, self.reduction, self.eps = name, reduction, eps

    def forward(self, prediction, target, context):
        error = prediction - target
        if self.name in {"nmse", "nmae"}:
            _, scale = normal_stats(context)
            error = error / (scale + self.eps)
        elif self.name == "rmse":
            scale = context.mean(-1, keepdim=True).abs()
            error = error / (scale + self.eps)
        loss = error.abs() if self.name in {"mae", "nmae"} else error.square()
        return loss.mean() if self.reduction == "mean" else loss


def make_losses(training_loss):
    names = ["mse", "nmse", "mae", "nmae", "rmse"]
    return ForecastLoss(training_loss), {name: ForecastLoss(name, "none") for name in names}


class TorchLearner:
    def __init__(self, model, criterion, eval_losses, lr, device):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "gpu":
            device = "cuda"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = criterion
        self.eval_losses = eval_losses
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def fit(self, train_loader, valid_loaders, epochs, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        history = {"train": [], "valid": {name: [] for name in valid_loaders}}
        for _ in range(epochs):
            self.model.train()
            epoch_losses = []
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(x), y, x)
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())
            history["train"].append(float(np.mean(epoch_losses)))
            for name, loader in valid_loaders.items():
                history["valid"][name].append(self.evaluate(loader, keep_all=False))
        return history

    def evaluate(self, loader, keep_all=True):
        losses = {name: [] for name in self.eval_losses}
        self.model.eval()
        with torch.inference_mode():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                prediction = self.model(x)
                for name, loss in self.eval_losses.items():
                    losses[name].append(loss(prediction, y, x).cpu())
        losses = {name: torch.cat(values) for name, values in losses.items()}
        return losses if keep_all else {name: value.mean().item() for name, value in losses.items()}
