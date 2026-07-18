# RevIN experiment

This independent experiment studies **when instance normalization is useful**
for univariate time-series forecasting. It first reproduces the earlier RevIN
ablation after the code rewrite, then extends it across datasets, settings, and
the DLinear/PatchTST backbones. Proposed extensions such as cmIN belong in a
separate future experiment and are intentionally not implemented here.

## Layout

```text
src/
  conf/config.yaml
  models/                 DLinear and PatchTST
  scripts/experiment.py   train/evaluate one configuration (or a seed list)
  slurm/*.sh              configuration enumeration, execution, and tables
  utils/                  data, normalization, losses, training, plots, tables
  tests/                  lightweight checks
latex/ECML_submission/    paper source
latex/experiment_guides/  concise experiment protocols
datasets/                 optional repo-local remote datasets
weights/                  placeholder; unused by the current backbones
outputs/                  runs, figures, metrics, summaries, and tables
logs/                     Slurm/runtime logs
revin.slurm               only Slurm submission file
```

Each dataset is read from `datasets/<name>/<name>.csv`; the first column is the
date index and the remaining columns are user series. Dates are split
chronologically. With `data.indiv_split<1`, seen and unseen users produce
`valid1/test1` and `valid2/test2` splits.

Training already uses random user/window sampling. Evaluation enumerates windows
with `data.eval_stride`, which defaults to the forecast horizon in the launcher.
This avoids the former individual-ID sampling cost and highly overlapping test
windows without introducing an unsupported sampling option.

## One configuration

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

`seeds` expands a configuration into isolated `seed_N/` runs. Each run saves its
resolved config, model, history, criterion plot, losses, and JSON summary. The
JSON reports the mean, population standard deviation, population variance, and
count of per-point loss contributions. Training histories contain raw step
losses, interval-average train losses, and validation metrics at the same
optimizer steps.

Losses are MSE, MAE, normalized MSE/MAE, and relative MSE. The legacy `rmse`
name remains accepted as an alias for relative MSE. The benchmark methods are
no/global normalization, non-affine instance normalization, affine RevIN,
data-space versus normalized-space backpropagation, last-value centering, and
the arcsinh transform.

## Required order on Slurm

First run the dependency-light local check in the prepared environment:

```bash
python src/tests/results_test.py
python src/tests/smoke_test.py
```

Then submit the benchmark smoke gate:

```bash
TEST_MODE=true sbatch revin.slurm
```

Test mode keeps the previous Electricity/Solar, `168:24`/`720:168`, PatchTST,
and seeds 1--2 sweep, but defaults to 20 epochs with validation every 10 steps.
It writes to `outputs/revin_experiment_test`, so it cannot overwrite or pollute
the publication sweep. Inspect all `seed_N/results.json`, histories, plots, and
the generated tables before continuing. Increase `EPOCHS` if 20 steps do not
exercise validation on the cluster.

Normal mode defaults to:

- datasets: ETTh1, Electricity, Traffic, Solar, Weather, Exchange Rate;
- settings: `168:24`, `168:168`, `504:24`, `504:168`, `504:504`, `720:168`,
  `720:720`;
- models: DLinear and PatchTST;
- methods: all eight benchmark and appendix methods;
- seeds: 1--5, with 10,000 epochs and validation/logging every 1,000 steps.

The complete Cartesian product is too large for one sequential 23-hour job.
The default sweep contains 672 dataset/setting/model/method configurations, so
use one configuration per array element and build tables only after every task
succeeds. For example, with at most 24 tasks running concurrently:

```bash
train_job=$(sbatch --parsable --array=0-671%24 \
  --export=ALL,TEST_MODE=false,RUN_MODE=train,SHARD_COUNT=672 \
  revin.slurm)

sbatch --dependency=afterok:$train_job \
  --export=ALL,TEST_MODE=false,RUN_MODE=tables \
  revin.slurm
```

Replace the `%24` throttle with a value allowed by the cluster quota. With sweep
overrides, set `SHARD_COUNT` and the inclusive array range to the resulting
configuration count (`datasets * settings * models * methods`); reproduction
subsets therefore use their own matching count. A shard selects configurations
deterministically by `configuration_index % SHARD_COUNT`; each selected configuration
still runs all requested seeds. `RUN_MODE=both` remains convenient for smoke or
a deliberately narrowed unsharded sweep. Set `SKIP_COMPLETED=true` only when
resuming an otherwise identical sweep.

## Sweep overrides

The launcher accepts space- or comma-separated environment overrides:

```bash
DATASETS="electricity traffic" \
SETTINGS="168:24 504:168" \
MODELS=patchtst \
METHODS="none_mse standard_mse instance_mse instance_nmse revin_mse revin_nmse" \
SEEDS="1 2 3" \
EPOCHS=10000 VALID_EVAL_FREQ=1000 LOGGING_EVAL_FREQ=1000 \
EVAL_STRIDE=horizon TEST_MODE=false RUN_MODE=both \
sbatch revin.slurm
```

Other controls are `BATCH_SIZE`, `LEARNING_RATE`, `OUT_ROOT`, `DATA_ROOT`,
`VENV_ACTIVATE`, `SUMMARY_METHODS`, `ORACLE_METHODS`, `BASELINE_METHOD`,
`GENERATE_SUMMARY`, and `STRICT_SUMMARY`. `EVAL_STRIDE` may be `horizon` or a
positive integer.

If `DATA_ROOT` is unset, the launcher searches for each CSV under the repository
`datasets/`, its parent `datasets/`, and one additional shared-parent candidate.
Set `DATA_ROOT=/cluster/path/to/datasets` when the checkout lives
elsewhere. The active models do not read pretrained weights.

## Executable files

- `revin.slurm` is the only file submitted with `sbatch`. Edit its
  partition, time limit, resources, `TEST_MODE`, and `RUN_MODE`.
- `src/slurm/run_revin_experiment.sh` resolves data, enumerates the requested
  configurations, launches one Python process per configuration, and builds
  aggregate tables. It logs every parameter that distinguishes adjacent runs.
- `src/scripts/experiment.py` is the Hydra training/evaluation entry point. It
  expands `seeds`, writes one isolated directory per seed, and timestamps the
  training and validation messages.
- `src/utils/results.py` validates completed seed outputs and creates LaTeX and
  JSON summaries. It is normally called by table mode rather than directly.

Timestamped progress, validation, evaluation, and table messages are written
to the Slurm `.out` file. The `.err` file is reserved for warnings and errors.

## Result interpretation

For every model and test split, table mode writes a row-wise LaTeX table with
seed mean $\pm$ sample standard deviation and an explicit per-row
`\times 10^m` multiplier. It also writes `summary_*.json` and `summary_*.tex`
for global standardization, non-affine instance normalization, affine RevIN,
and an oracle that chooses between global standardization and full affine RevIN
using the lowest mean test MSE per dataset/setting.

The summaries give every dataset/setting equal weight, use only seeds complete
for every compared method, report seed-level macro standard deviation and
variance, and retain the mean within-run loss variance. They also report mean
per-setting relative improvement, which is more interpretable than raw MSE when
datasets have different units. The test-selected oracle is deliberately
optimistic and is only a reference for the potential value of choosing
normalization per setting; it must not be presented as a deployable policy.

The energy-distance and t-SNE diagnostics are computed separately in the
`dataset_visu` notebook and may be imported into the paper after the forecasting
results are reproduced. They are not part of this training launcher.

## Experiment guides

The one-page protocols under `latex/experiment_guides/` cover normalization
components, normalized backpropagation, and centering/transform appendix runs.
Their PDFs are kept beside the sources; ignored convenience copies may also be
written to `outputs/pdf/`.
