# RevIN experiment

This independent experiment studies **when instance normalization is useful**
for univariate time-series forecasting. It first reproduces the earlier RevIN
ablation after the code rewrite, then extends it across datasets, settings, and
the DLinear/PatchTST backbones. Global modulated instance normalization (MIN)
is included as the non-personalized asymmetric-output precursor to cmIN.
Cluster personalization and data-dependent cmIN initialization remain future
work.

## Layout

```text
src/
  conf/config.yaml
  models/                 DLinear and PatchTST
  scripts/experiment.py   train/evaluate one configuration (or a seed list)
  slurm/run_*.sh          complete workflow orchestration
  slurm/stage_*.sh        separate training and table stages
  utils/                  data, normalization, losses, training, plots, tables
  tests/                  lightweight checks
latex/ECML_submission/    paper source
latex/experiment_guides/  concise experiment protocols
datasets/                 tracked configs and optional repo-local CSVs
weights/                  placeholder; unused by the current backbones
outputs/                  runs, figures, metrics, summaries, and tables
logs/                     Slurm/runtime logs
revin.slurm               core methods and the test smoke gate
nmse.slurm                no/standard normalization under nMSE
exotic.slurm              component and transform ablations
min.slurm                 global MIN under MSE and nMSE
```

Each dataset is read from `datasets/<name>/<name>.csv`; the first column is the
date index and the remaining columns are user series. Dates are split
chronologically. With `data.indiv_split<1`, seen and unseen users produce
`valid1/test1` and `valid2/test2` splits.

Every sampled integer `t` is the last observed date. A forecasting pair is
`X_t = X(t-L:t] = {x_(t-L+1), ..., x_t}` and
`Y_t = X(t:t+H] = {x_(t+1), ..., x_(t+H)}`. Chronological splits own target
dates: the full horizon must stay inside its split, while the lookback may
cross the preceding boundary. Thus the first validation/test query is the date
immediately before that target period, and changing `L` does not change which
validation/test target dates are evaluated.

A sibling `datasets/<name>/config.json` is loaded automatically. Shared
`drop_users` entries are zero-based positions among the original value columns;
additional RevIN-only exclusions may be placed under a `revin` object. The two
lists and any `data.drop_users` Hydra additions are merged, so a run cannot
silently re-enable a dataset-level exclusion. A portable `target_cols` list may
select named variables; the project-scoped value overrides the shared value and
an explicit `data.target_cols` run value overrides both. ETTh1, ETTh2, ETTm1,
and ETTm2 use every non-date variable in their source CSVs; cluster copies must therefore be the
complete seven-variable file. Set `data.config_path` only to use an explicit
JSON file or directory. Portable `date_col`, `aggr`, and `aggr_period` fields
are also applied; the shared configs aggregate Solar to hourly sums and Weather
to hourly means.
Every seed output records the `query_t` window anchor, effective path, applied
indices, selected target columns, aggregation settings, dropped column names,
and retained-user count in `dataset_config.json`.

The repository tracks the curated Electricity configuration while leaving its
CSV ignored. It includes every currently identified user with a constant run of
at least 168 samples, including source column 245 found by the smoke audit.

The shared Weather configuration retains every variate, including columns 4,
7, 14, and 15, because the panel is already small. Their exact constant input
windows remain known diagnostics from the stride-one `168:24` audit rather
than dataset-level exclusions. ETT and Exchange Rate likewise retain all
non-date variates.

This is a curated exclusion list, not automatic constant-window detection. A
user omitted from the JSON may still contain a constant look-back and destabilize
nMSE; the constant-user policy should be evaluated separately before the full
normalized-loss sweep.

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
  training.loss=nmse training.epochs=10000 training.steps=10000 \
  training.valid_eval_freq=1000 training.logging_eval_freq=1000 \
  seeds='[1,2,3]' output.name=patchtst_revin_nmse
```

`seeds` expands a configuration into isolated `seed_N/` runs. Each run saves its
resolved config, applied dataset config, model, history, criterion plot, losses,
and JSON summary. The
JSON reports the mean, population standard deviation, population variance, and
count of per-point loss contributions. Training histories contain raw step
losses, interval-average train losses, and validation metrics at the same
optimizer steps.

Losses are MSE, MAE, normalized MSE/MAE, and relative MSE. The legacy `rmse`
name remains accepted as an alias for relative MSE. The Slurm study separates
distinct scientific questions into four fronts. `revin.slurm` compares no
normalization, global standardization, non-affine instance normalization, and
affine RevIN. `nmse.slurm` isolates the no-normalization and global-standard
nMSE controls. `exotic.slurm` contains mean-only, scale-only, median/MAD,
last-value, and arcsinh variants. `min.slurm` isolates global MIN. The launcher
intentionally does not schedule interaction-only compositions such as
last-value centering plus arcsinh plus affine parameters.

`RevIN` accepts independent `center=mean|last|median|none` and
`scale=std|mad|none` strategies. MAD is the raw per-window median absolute
deviation around the window median. Mean-only uses `center=mean,scale=none`;
scale-only uses `center=none,scale=std`; and the robust variant uses
`center=median,scale=mad`.

`MIN` retains RevIN's input affine parameters but unties the output:

```text
x_tilde = gamma * (x - location) / scale + nu
y_hat   = scale * (alpha * (f(x_tilde) - nu) / gamma + beta) + location
```

`gamma`, `nu`, `alpha`, and `beta` are global per-channel parameters initialized
to the identity. Unlike cmIN, the current MIN has no user/cluster-conditioned
parameters and no empirical delta/lambda initialization.

## Complete Slurm workflow

First run the dependency-light local check in the prepared environment:

```bash
python src/tests/results_test.py
python src/tests/test_slurm_workflow.py
python src/tests/smoke_test.py
```

Then submit the benchmark smoke gate:

```bash
EXPERIMENT_MODE=test sbatch revin.slurm
```

Test mode uses only Electricity at `504:168`, PatchTST, and seed 1. It runs
`standard_mse`, `instance_mse`, `instance_nmse`, and `min_nmse`: 4
configurations and 4 seed-runs. Each run uses exactly 2,000 optimizer steps
with validation and progress logging every 200 steps. The `v2` completion
signature deliberately invalidates the previous eight-method smoke outputs,
so this reduced gate must be rerun once. Outputs go to
`outputs/revin_experiment_test`, so they cannot overwrite or pollute the
publication sweep. Inspect all `seed_N/results.json`, histories, plots, and the
generated tables before continuing.

Every front defaults to the ordered `STAGES=train,tables` workflow.
The orchestrator in `src/slurm/run_revin_experiment.sh` resolves the scale and
scientific grid; `stage_train.sh` fits only missing or stale configurations and
`stage_tables.sh` requires the full selected grid before aggregating it.
`STAGES=train` or `STAGES=tables` is available only as a recovery override.

The publication profiles share the same five settings (`168:24`, `336:48`,
`504:168`, `336:96`, and `336:720`), seeds 1--3, and exactly 10,000 optimizer
steps. They differ only in datasets and models:

- `small`: Traffic, Electricity, and Solar with PatchTST.
- `full`: all small datasets plus Weather, Exchange Rate, ETTh1, ETTh2, ETTm1,
  and ETTm2 with PatchTST.
- `ultra`: the full grid with DLinear and PatchTST.

The fronts contribute these disjoint method families to the shared
`outputs/revin_experiment` run tree:

| Front | Methods | Small configurations / seeds | Full configurations / seeds | Ultra configurations / seeds |
|---|---|---:|---:|---:|
| `revin.slurm` | `none_mse`, `standard_mse`, `instance_mse`, `instance_nmse`, `revin_mse`, `revin_nmse` | 90 / 270 | 270 / 810 | 540 / 1,620 |
| `nmse.slurm` | `none_nmse`, `standard_nmse` | 30 / 90 | 90 / 270 | 180 / 540 |
| `exotic.slurm` | mean-only, scale-only, median/MAD, last-value, and arcsinh, each under MSE/nMSE | 150 / 450 | 450 / 1,350 | 900 / 2,700 |
| `min.slurm` | `min_mse`, `min_nmse` | 30 / 90 | 90 / 270 | 180 / 540 |

Run the core study first, then extend it:

```bash
EXPERIMENT_MODE=small sbatch revin.slurm
EXPERIMENT_MODE=full sbatch revin.slurm
EXPERIMENT_MODE=ultra sbatch revin.slurm
```

Run the other families later by replacing the front, for example
`EXPERIMENT_MODE=small sbatch nmse.slurm`. Small can safely precede full, then
ultra. Every mode defaults to `SKIP_COMPLETED=true`. The per-seed signature does
not contain the mode or family and hashes dataset-config content rather than
its modification time: a full run therefore skips the exact small
dataset/setting/model/method/seed results and trains only the added datasets;
ultra similarly adds only DLinear after full. A seed is reusable only when
`results.json`, its resolved configuration and dataset provenance, and its
signature-matched `run.complete` are current. Each family owns
`.workflow/<family>/train.complete`, `.workflow/<family>/tables.complete`, and
`tables/<family>/` outputs. Set `SKIP_COMPLETED=false` to force the selected
workflow. If a sequential allocation exceeds the time limit, resubmit the same
command; completed seeds are skipped.

Tables requested with seed standard deviations remain valid in test mode: a
single seed has no estimable sample standard deviation, so test cells show the
value without `±`; macro uncertainty fields use `--`. Publication modes retain
their multi-seed estimates. The table-stage `v3` signature rebuilds prior
tables automatically without retraining completed runs.

## Sweep overrides

The launcher accepts space- or comma-separated environment overrides:

```bash
DATASETS="electricity traffic" \
SETTINGS="168:24 504:168" \
MODELS=patchtst \
METHODS="mean_only_nmse scale_only_nmse median_mad_nmse revin_last_nmse revin_arcsinh_nmse" \
SEEDS="1 2 3" \
EPOCHS=10000 STEPS=10000 VALID_EVAL_FREQ=1000 LOGGING_EVAL_FREQ=1000 \
EVAL_STRIDE=horizon EXPERIMENT_MODE=full STAGES=train,tables \
sbatch exotic.slurm
```

Other controls are `BATCH_SIZE`, `LEARNING_RATE`, `EPOCHS`, `STEPS`, `OUT_ROOT`, `DATA_ROOT`,
`VENV_ACTIVATE`, `SUMMARY_METHODS`, `ORACLE_METHODS`, `BASELINE_METHOD`,
`GENERATE_SUMMARY`, and `STRICT_SUMMARY`. `EVAL_STRIDE` may be `horizon` or a
positive integer.

If `DATA_ROOT` is unset, the launcher searches for each CSV under the repository
`datasets/`, its parent `datasets/`, and one additional shared-parent candidate.
Set `DATA_ROOT=/cluster/path/to/datasets` when the checkout lives
elsewhere. The active models do not read pretrained weights.

## Executable files

- `revin.slurm`, `nmse.slurm`, `exotic.slurm`, and `min.slurm` are the four
  submission fronts. Each selects one method family and accepts
  `EXPERIMENT_MODE=small|full|ultra`; `revin.slurm` additionally owns `test`.
- `src/slurm/run_revin_experiment.sh` resolves data, enumerates the requested
  configurations, validates `STAGES`, and orchestrates the complete workflow.
- `src/slurm/stage_train.sh` and `stage_tables.sh` execute the separate stages;
  the root front sources them and users do not submit them directly.
- `src/scripts/experiment.py` is the Hydra training/evaluation entry point. It
  expands `seeds`, writes one isolated directory per seed, and timestamps the
  training and validation messages.
- `src/utils/results.py` validates completed seed outputs and creates LaTeX and
  JSON summaries. It is normally called by the table stage rather than directly.

Timestamped progress, validation, evaluation, table messages, and Python
warnings are written to the Slurm `.out` file. The `.err` file is reserved for
scheduler, shell, or Python failures.

## Result interpretation

For every family, model, and test split, the table stage writes
`tables/<family>/results_<model>_<split>_mse.tex` with seed mean $\pm$ sample
standard deviation and an explicit per-row `\times 10^m` multiplier. It also
writes `summary_*.json` and `summary_*.tex` for every method scheduled by that
front. By default, the validation-selected policy and
optimistic test oracle choose among all those methods. `SUMMARY_METHODS` and
`ORACLE_METHODS` can narrow the comparison when a specific paper table requires
fewer candidates.

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
