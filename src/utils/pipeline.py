"""Losses and the training/evaluation loop."""

import logging
import random

import numpy as np
import torch
from torch import nn

from utils.models import normal_stats


LOGGER = logging.getLogger(__name__)


class ForecastLoss(nn.Module):
    def __init__(self, name="nmse", reduction="mean", eps=1e-8):
        super().__init__()
        self.name, self.reduction, self.eps = name, reduction, eps

    def forward(self, prediction, target, context):
        error = prediction - target
        if self.name in {"nmse", "nmae"}:
            _, scale = normal_stats(context)
            error = error / (scale + self.eps)
        elif self.name in {"rmse", "relative_mse"}:
            scale = context.mean(-1, keepdim=True).abs()
            error = error / (scale + self.eps)
        loss = error.abs() if self.name in {"mae", "nmae"} else error.square()
        return loss.mean() if self.reduction == "mean" else loss


def make_losses(training_loss):
    names = ["mse", "nmse", "mae", "nmae", "relative_mse"]
    if training_loss not in names:
        names.append(training_loss)
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

    def fit(
        self,
        train_loader,
        valid_loaders,
        epochs,
        seed,
        valid_eval_freq=None,
        logging_eval_freq=None,
        steps=None,
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        default_freq = max(1, len(train_loader))
        valid_eval_freq = int(valid_eval_freq or default_freq)
        logging_eval_freq = int(logging_eval_freq or valid_eval_freq)
        if valid_eval_freq < 1 or logging_eval_freq < 1:
            raise ValueError("evaluation frequencies must be positive")
        if logging_eval_freq % valid_eval_freq:
            raise ValueError("logging_eval_freq must be a multiple of valid_eval_freq")
        history = {
            "train": [],
            "train_batch": [],
            "valid": {name: [] for name in valid_loaders},
        }
        recent_losses = []
        step = 0
        max_steps = None if steps is None else int(steps)
        if max_steps is not None and max_steps < 1:
            raise ValueError("steps must be positive")

        def evaluate_interval(log=False):
            if not recent_losses:
                return
            train_loss = float(np.mean(recent_losses))
            history["train_batch"].append(
                {"step": step, "loss": train_loss, "losses": {self.criterion.name: train_loss}}
            )
            recent_losses.clear()
            valid_results = {}
            for name, loader in valid_loaders.items():
                values = self.evaluate(loader, keep_all=False)
                history["valid"][name].append({"step": step, "losses": values})
                valid_results[name] = values
            if log:
                LOGGER.info(
                    "step=%s train_interval=%.6g valid=%s", step, train_loss, valid_results
                )

        epoch = 0
        while (
            (max_steps is None and epoch < epochs)
            or (max_steps is not None and step < max_steps)
        ):
            epoch += 1
            self.model.train()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(x), y, x)
                loss.backward()
                self.optimizer.step()
                value = loss.item()
                step += 1
                history["train"].append(value)
                recent_losses.append(value)
                if step % valid_eval_freq == 0:
                    evaluate_interval(log=step % logging_eval_freq == 0)
                if max_steps is not None and step >= max_steps:
                    break
        evaluate_interval(log=True)
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
