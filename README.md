# RevIN experiments

Compact research code for the RevIN forecasting experiments. The sweep trains
the original PatchTST and DLinear architectures from scratch across datasets,
lookback-horizon settings, normalization variants, and seeds.

## Layout

```text
src/
  run_revin_experiment.slurm   complete Slurm sweep
  conf/config.yaml       default experiment configuration
  models/                PatchTST and DLinear implementations
  utils/                 data, normalization, training, and results code
  scripts/experiment.py  Hydra training and evaluation entry point
  tests/                 lightweight smoke test
datasets/<name>/<name>.csv
```

Each CSV is dates by users. The first column is ignored as the row index; all
remaining columns are numeric user series. `data.drop_users` contains column
indexes to remove. Data are split chronologically into train, validation, and
test periods. `data.indiv_split=1` gives three splits; a smaller value also
splits users into seen and unseen groups, giving six splits.

Training uses random windows, Adam, batch size 256, and learning rate `1e-5`.
Evaluation visits every window deterministically with `data.eval_stride`.
Normalization is `none`, global `standard`, or `revin`; RevIN behavior is set
through `normalization.kwargs` (`affine`, `center`, and `transform`).

## Run

Create the environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Launch the complete experiment from the repository root:

```bash
sbatch src/run_revin_experiment.slurm
```

The script runs both models, all normalization methods, datasets, L-H settings,
and seeds. It writes each run under `outputs/revin_experiment/`, then creates one TeX
table per test split. Set `SHOW_STD=false` in the script to omit `\pm` standard
deviations across seeds.

A lightweight local check is available with:

```bash
python src/tests/smoke_test.py
```
