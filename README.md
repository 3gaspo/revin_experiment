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
datasets/                 tracked configs and optional repo-local CSVs
weights/                  placeholder; unused by the current backbones
outputs/                  runs, figures, metrics, summaries, and tables
logs/                     Slurm/runtime logs
revin.slurm               only Slurm submission file
```

Each dataset is read from `datasets/<name>/<name>.csv`; the first column is the
date index and the remaining columns are user series. Dates are split
chronologically. With `data.indiv_split<1`, seen and unseen users produce
`valid1/test1` and `valid2/test2` splits.

A sibling `datasets/<name>/config.json` is loaded automatically. Shared
`drop_users` entries are zero-based positions among the original value columns;
additional RevIN-only exclusions may be placed under a `revin` object. The two
lists and any `data.drop_users` Hydra additions are merged, so a run cannot
silently re-enable a dataset-level exclusion. A portable `target_cols` list may
select named variables; the project-scoped value overrides the shared value and
an explicit `data.target_cols` run value overrides both. ETTh1 is configured as
OT-only. Set `data.config_path` only to use an explicit JSON file or directory.
Every seed output records the effective path, applied indices, selected target
columns, dropped column names, and retained-user count in
`dataset_config.json`.

The repository tracks the curated Electricity configuration while leaving its
CSV ignored. It includes every currently identified user with a constant run of
at least 168 samples, including source column 245 found by the smoke audit.

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
EXPERIMENT_MODE=test sbatch revin.slurm
```

Test mode uses Electricity/Solar, `168:24`/`720:168`, PatchTST, and seeds 1--2.
It compares only global standardization and non-affine instance normalization,
each with MSE and nMSE loss: 16 configurations in total. Each run uses exactly
2,000 optimizer steps with validation and progress logging every 200 steps. Outputs go to
`outputs/revin_experiment_test`, so they cannot overwrite or pollute the
publication sweep. Inspect all `seed_N/results.json`, histories, plots, and the
generated tables before continuing.

The publication profiles share `outputs/revin_experiment`:

- `small`: Traffic, Electricity, and Solar; `168:24`, `504:24`, `504:168`, and
  `504:504`; PatchTST; the
  six core methods (`none_mse`, `standard_mse`, `instance_mse`,
  `instance_nmse`, `revin_mse`, `revin_nmse`); seeds 1--3; exactly 10,000
  optimizer steps. This is 72 configurations and 216 seed-runs.
- `full`: ETTh1 (OT only), Electricity, Traffic, Solar, Weather, and Exchange
  Rate; the four small settings plus `336:96` and `336:720`; PatchTST; all nine
  methods, including `standard_nmse`, last-value centering, and arcsinh; seeds
  1--3. This is 324 configurations and 972 seed-runs.
- `ultra`: the full profile with both DLinear and PatchTST. This is 648
  configurations and 1,944 seed-runs.

`large` remains accepted as a compatibility alias for `full`.

Run the core study first, then extend it:

```bash
EXPERIMENT_MODE=small sbatch revin.slurm
EXPERIMENT_MODE=full sbatch revin.slurm
EXPERIMENT_MODE=ultra sbatch revin.slurm
```

Yes: small can safely precede full, then ultra. All publication profiles default to
`SKIP_COMPLETED=true`. Before each configuration, the launcher checks every
requested `seed_N`; seeds with a non-empty result, resolved config containing
the requested step budget, and dataset provenance older than the result are
reused. Large therefore computes only missing small seeds plus its new
datasets, DLinear runs, and additional methods. Test defaults to
`SKIP_COMPLETED=false` because its isolated 2,000-step outputs are intended to
be refreshed. Set `SKIP_COMPLETED=false` explicitly after changing another
training hyperparameter. If a sequential allocation exceeds the time limit,
resubmit the same profile; split first by model and then dataset only when
needed. `RUN_MODE=train` and `RUN_MODE=tables` can separate computation from
aggregation.

## Sweep overrides

The launcher accepts space- or comma-separated environment overrides:

```bash
DATASETS="electricity traffic" \
SETTINGS="168:24 504:168" \
MODELS=patchtst \
METHODS="none_mse standard_mse instance_mse instance_nmse revin_mse revin_nmse" \
SEEDS="1 2 3" \
EPOCHS=10000 STEPS=10000 VALID_EVAL_FREQ=1000 LOGGING_EVAL_FREQ=1000 \
EVAL_STRIDE=horizon EXPERIMENT_MODE=full RUN_MODE=both \
sbatch revin.slurm
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

- `revin.slurm` is the only file submitted with `sbatch`. Edit its
  partition, time limit, resources, `EXPERIMENT_MODE`, and `RUN_MODE`.
- `src/slurm/run_revin_experiment.sh` resolves data, enumerates the requested
  configurations, launches one Python process per configuration, and builds
  aggregate tables. It logs every parameter that distinguishes adjacent runs.
- `src/scripts/experiment.py` is the Hydra training/evaluation entry point. It
  expands `seeds`, writes one isolated directory per seed, and timestamps the
  training and validation messages.
- `src/utils/results.py` validates completed seed outputs and creates LaTeX and
  JSON summaries. It is normally called by table mode rather than directly.

Timestamped progress, validation, evaluation, table messages, and Python
warnings are written to the Slurm `.out` file. The `.err` file is reserved for
scheduler, shell, or Python failures.

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
