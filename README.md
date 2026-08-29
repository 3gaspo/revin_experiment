# RevIN experiment

This project studies when reversible instance normalization helps univariate
forecasting. It separates normalization from normalized loss, decomposes
centering and scaling choices, and evaluates global modulated instance
normalization (MIN) with pinned DLinear and PatchTST implementations.

The previously migrated configurations predate the pinned backbone packages.
They are historical only; current comparisons require new completed runs.

## Documentation map

| Need | Document |
|---|---|
| Paper-ready RevIN and MIN formulation | [`latex/method_overview.pdf`](latex/method_overview.pdf) |
| Normalization, backbone, and reporting flow | [`docs/architecture.md`](docs/architecture.md) |
| Four Slurm families and rerun order | [`docs/experiment_catalog.md`](docs/experiment_catalog.md) |
| Current versus superseded evidence | [`docs/results_recap.md`](docs/results_recap.md) |
| Complete reproducibility specification | [`latex/experiment_guideline.pdf`](latex/experiment_guideline.pdf) |
| Full historical and analyzed evidence record | [`latex/executive_summary.pdf`](latex/executive_summary.pdf) |

## Setup

Use the project-managed environment from the repository root:

```bash
uv sync
export PYTHONPATH=src
```

Place each wide CSV under `datasets/<name>/`; an adjacent `config.json`
selects targets, exclusions, date handling, aggregation, and missing values.
`missing_values` defaults to `zero`; `error` rejects CSV NaNs, and infinite
values are always rejected. DLinear and
PatchTST train locally within each run and require no checkpoint payload.

## Main executions

Establish the core reference before the focused extensions:

```bash
EXPERIMENT_MODE=test sbatch revin.slurm
EXPERIMENT_MODE=full sbatch revin.slurm
EXPERIMENT_MODE=full sbatch nmse.slurm
EXPERIMENT_MODE=full sbatch exotic.slurm
EXPERIMENT_MODE=full sbatch min.slurm
```

`revin.slurm` compares no, global, instance, and affine reversible
normalization. `nmse.slurm` isolates normalized-loss controls;
`exotic.slurm` decomposes centering, scaling, and transforms; `min.slurm`
tests the asymmetric output mapping. The exact methods and publication grids
are in the [experiment catalog](docs/experiment_catalog.md).

Each front defaults to `STAGES=train,tables`. Resubmitting the same command is
the normal recovery procedure; a stage subset is an explicit recovery
override. Matching Selena fronts keep the same science:

```bash
EXPERIMENT_MODE=test sbatch revin_selena.slurm
```

## Outputs and cluster operations

- Family runs: `outputs/<core|nmse|exotic|min>/`.
- Aggregate reports: `outputs/reports/<family>/<mode>/`.
- Seed artifacts: `run_n/seed_n/` below each scientific identity.
- Runtime streams: `logs/` on DGX and `logs_selena/` on Selena.

Preview and then mirror maintained code from DGX:

```bash
bash sync_code_to_selena.sh --dry-run
bash sync_code_to_selena.sh
```

The dry run itemizes additions, updates, and `*deleting` stale-code entries.
Delayed deletion preserves excluded environments, dependency manifests,
datasets, weights, outputs, and logs.

Pull Selena results from DGX at the smallest useful depth:

```bash
bash sync_results_to_dgx.sh
bash sync_results_to_dgx.sh --size detailed
bash sync_results_to_dgx.sh --size full
```

The default retrieves logs and aggregate reports; `detailed` adds non-binary
run diagnostics and `full` adds binary recovery payloads. Use
`bash publish_job.sh <job-id>` for one terminal log pair or
`bash publish_job.sh` for all logs plus aggregate reports.

## Documentation maintenance

```bash
PYTHONPATH=src python -m scripts.build_docs
PYTHONPATH=src python -m scripts.build_docs --render method
PYTHONPATH=src python -m scripts.build_docs --render all
```

The default validates the documentation map and all DGX fronts. Mathematical
changes belong in the method note and guideline, package-boundary changes in
architecture, planned comparisons in the catalog, and analyzed evidence in
the recap and executive summary.
