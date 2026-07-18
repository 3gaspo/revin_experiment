"""Train and evaluate one forecasting configuration."""

import json
import logging
import sys
from pathlib import Path
from time import perf_counter

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from utils.dataset import build_loaders
from utils.models import build_model
from utils.pipeline import TorchLearner, make_losses
from utils.plots import plot_history


def summarize(losses):
    return {
        split: {
            metric: {
                "mean": values.float().mean().item(),
                "std": values.float().std(unbiased=False).item(),
                "variance": values.float().var(unbiased=False).item(),
                "count": values.numel(),
            }
            for metric, values in metrics.items()
        }
        for split, metrics in losses.items()
    }


def run_experiment(cfg: DictConfig):
    started = perf_counter()
    output = Path(cfg.output.dir) / cfg.output.name
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / "config.yaml", resolve=True)

    loaders, stats, dim = build_loaders(
        cfg.data, cfg.task.lags, cfg.task.horizon, cfg.training.batch_size, cfg.seed
    )
    model = build_model(
        cfg.model, cfg.normalization, cfg.task.lags, cfg.task.horizon, dim, stats
    )
    criterion, eval_losses = make_losses(cfg.training.loss)
    learner = TorchLearner(
        model, criterion, eval_losses, cfg.training.lr, cfg.training.device
    )

    logging.info(
        "training %s/%s on %s -> %s",
        cfg.model.name,
        cfg.normalization.name,
        learner.device,
        output,
    )
    history = learner.fit(
        loaders["train"],
        {name: loader for name, loader in loaders.items() if name.startswith("valid")},
        cfg.training.epochs,
        cfg.seed,
        cfg.training.get("valid_eval_freq"),
        cfg.training.get("logging_eval_freq"),
    )
    model.save(output / "model.pt")
    torch.save(history, output / "history.pt")
    plot_history(
        history,
        cfg.training.loss,
        output / "criterion_loss.pdf",
        bool(cfg.training.get("plot_step_train_loss", False)),
    )

    losses = {
        name: learner.evaluate(loader)
        for name, loader in loaders.items()
        if name.startswith("test")
    }
    torch.save(losses, output / "losses.pt")
    (output / "results.json").write_text(json.dumps(summarize(losses), indent=2))
    logging.info("finished in %.1f seconds", perf_counter() - started)


def run_experiments(cfg: DictConfig):
    seeds = cfg.get("seeds")
    if not seeds:
        return {int(cfg.seed): run_experiment(cfg)}
    base_name = str(cfg.output.name).rstrip("/")
    results = {}
    for seed in seeds:
        seeded = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        seeded.seed = int(seed)
        seeded.seeds = None
        seeded.output.name = f"{base_name}/seed_{int(seed)}"
        results[int(seed)] = run_experiment(seeded)
    return results


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    run_experiments(cfg)


if __name__ == "__main__":
    main()
