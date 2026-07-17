# RevIN experiment

This is the compact, independent experiment extracted from TimeTensors for the normalization and normalized-loss study described in `latex/ECML_submission/`. It intentionally keeps a smaller data and training path while retaining the features needed to reproduce and extend the paper tables.

## Layout

```text
src/
  conf/config.yaml
  models/                 DLinear and PatchTST
  scripts/experiment.py   Hydra experiment entrypoint
  slurm/                  complete benchmark job
  utils/                  data, normalization, losses, training, plots, tables
  tests/                  lightweight smoke checks
latex/ECML_submission/    paper source
datasets/                 remote CSV datasets
weights/                  remote weights placeholder
outputs/                  models, histories, figures, metrics, and tables
logs/                     Slurm/runtime logs
```

Each dataset is read from `datasets/<name>/<name>.csv`; the first column is the date index and the remaining columns are user series. The chronological split is train/validation/test. With `data.indiv_split<1`, seen and unseen users produce `valid1/test1` and `valid2/test2` splits.

## Run

From the project root:

```bash
export PYTHONPATH=src
python -m scripts.experiment \
  data.name=electricity task.lags=168 task.horizon=24 \
  model.name=patchtst normalization.name=revin \
  training.loss=nmse training.epochs=10000 \
  training.valid_eval_freq=1000 training.logging_eval_freq=1000 \
  seeds='[1,2,3,4,5]' output.name=patchtst_revin_nmse
```

`seeds` expands a configuration into isolated `seed_N/` runs. Each run saves its resolved config, model, history, criterion plot, losses, and JSON summary. The training history contains raw step losses, the mean train loss over every validation interval, and validation metrics at the same optimizer steps. The criterion plot shows the interval-average train curve and validation curve; set `training.plot_step_train_loss=true` to add raw step losses.

Losses are MSE, MAE, normalized MSE/MAE, and relative MSE. The legacy `rmse` name remains accepted as an alias for relative MSE in old configurations. Normalization methods used by the paper are no normalization, global standardization, non-affine instance normalization, affine RevIN, last-value centering, and the arcsinh RevIN transform.

## Slurm benchmark

```bash
sbatch src/slurm/run_revin_experiment.slurm
TEST_MODE=true sbatch src/slurm/run_revin_experiment.slurm
```

Normal mode uses ETTh1, Electricity, Traffic, Solar, Weather, and Exchange Rate; settings 168--24, 168--168, 504--24, 504--168, 505--504, 720--168, and 720--720; DLinear and PatchTST; and five seeds. Test mode uses Electricity and Solar, settings 168--24 and 720--168, PatchTST only, and two seeds.

The job creates one table per model and test split. Tables report seed means and sample standard deviations with two decimals and an explicit per-row `\times 10^{m}` multiplier, allowing the paper tables to be regenerated after adding datasets.

## Lightweight check

With the prepared project environment:

```bash
python src/tests/smoke_test.py
```

The cmin and distribution-distance sections of the paper are retained as reference material but are outside the current experiment scope.

## Experiment guides

Concise one-page notes for normalization components, normalized
backpropagation, and centering/transform variants are under
`latex/experiment_guides/`, with each compiled PDF beside its `.tex` source.
Copies are also written to `outputs/pdf/`.
