# Experiment catalog

This page maps each front to one normalization question. Exact mathematical,
training, and artifact contracts remain in the
[`experiment guideline`](../latex/experiment_guideline.pdf).

## Shared publication grid

| Mode | Datasets | Settings | Backbones | Seeds |
|---|---|---|---|---|
| `test` | Electricity | `504:168` | PatchTST | 1 |
| `small` | Traffic, Electricity, Solar | Five declared settings | PatchTST | 1--3 |
| `full` | Small plus Weather, Exchange Rate, ETTh1/2, ETTm1/2 | `168:24`, `336:48`, `504:168`, `336:96`, `336:720` | PatchTST | 1--3 |
| `ultra` | Full profile | Full profile | PatchTST, DLinear | 1--3 |

## Slurm evaluations

| Front | Question | Compared models or ablations |
|---|---|---|
| `revin.slurm` | When does ordinary or affine instance normalization help? | none, global standard, instance, RevIN under MSE/nMSE |
| `nmse.slurm` | Is the gain due to normalized loss rather than instance normalization? | none and global standard under nMSE |
| `exotic.slurm` | Which centering, scaling, or transform component matters? | mean-only, scale-only, median/MAD, last-value, arcsinh |
| `min.slurm` | Does a global asymmetric output transform improve on RevIN? | MIN under MSE and nMSE |

Each front has a matching `*_selena.slurm` execution front with identical
science. Reports are written to `outputs/reports/<family>/<mode>/`.

## Recommended order

1. `revin.slurm` test, then small/full.
2. `nmse.slurm` to isolate loss effects.
3. `exotic.slurm` for component ablations.
4. `min.slurm` after the RevIN reference is established.
