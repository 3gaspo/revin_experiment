"""Build seed-aggregated tables and macro summaries from RevIN results."""

import argparse
import json
import logging
import math
import re
from collections import Counter
from itertools import product
from pathlib import Path
from statistics import fmean, stdev, variance


LOGGER = logging.getLogger(__name__)


def normalise_setting(setting):
    """Convert CLI forms such as ``168:24`` and ``168-24`` to ``168_24``."""
    return str(setting).replace(":", "_").replace("-", "_")


def discover_runs(root: Path, split: str, metric: str):
    """Return metric summaries indexed by dataset, setting, method, and seed."""
    runs = {}
    for path in sorted(root.rglob("results.json")):
        parts = path.relative_to(root).parts
        if len(parts) < 5:
            continue
        dataset, setting, method = parts[0], parts[1], "/".join(parts[2:-2])
        match = re.fullmatch(r"seed_(-?\d+)", parts[-2])
        if match is None:
            continue
        seed = int(match.group(1))
        summary = json.loads(path.read_text(encoding="utf-8"))[split][metric]
        std = float(summary.get("std", 0.0))
        record = {
            "mean": float(summary["mean"]),
            "std": std,
            "variance": float(summary.get("variance", std**2)),
            "count": int(summary.get("count", 0)),
        }
        runs.setdefault((dataset, setting, method), {})[seed] = record
    return runs


def discover(root: Path, split: str, metric: str):
    """Compatibility view used by the row-wise LaTeX table builder."""
    return {
        key: [record["mean"] for record in seed_records.values()]
        for key, seed_records in discover_runs(root, split, metric).items()
    }


def latex(text):
    return str(text).replace("_", r"\_")


def row_exponent(means):
    finite = sorted(abs(value) for value in means if math.isfinite(value) and value != 0)
    if not finite:
        return 0
    return math.floor(math.log10(finite[len(finite) // 2]))


def build_table(
    values, datasets, methods, split, metric, show_std, decimals=2, settings=None
):
    settings = {normalise_setting(setting) for setting in settings or []}
    rows = sorted({(dataset, setting) for dataset, setting, _ in values})
    rows = [
        row
        for row in rows
        if (not datasets or row[0] in datasets) and (not settings or row[1] in settings)
    ]
    methods = methods or sorted({method for _, _, method in values})
    uncertainty = " Seed uncertainty is the sample standard deviation." if show_std else ""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{metric.upper()} on {latex(split)}.{uncertainty}}}",
        rf"\begin{{tabular}}{{llc{'r' * len(methods)}}}",
        r"\toprule",
        "Dataset & $L$--$H$ & Scale & " + " & ".join(map(latex, methods)) + r" \\",
        r"\midrule",
    ]
    previous_dataset = None
    for dataset, setting in rows:
        means = {
            method: fmean(values[(dataset, setting, method)])
            for method in methods
            if values.get((dataset, setting, method))
        }
        if not means:
            continue
        best = min(means.values())
        exponent = row_exponent(means.values())
        divisor = 10.0**exponent
        cells = []
        for method in methods:
            seed_values = values.get((dataset, setting, method), [])
            if not seed_values:
                cells.append("--")
                continue
            mean = fmean(seed_values)
            cell = f"{mean / divisor:.{decimals}f}"
            if show_std and len(seed_values) > 1:
                cell += rf" $\pm$ {stdev(seed_values) / divisor:.{decimals}f}"
            if math.isclose(mean, best):
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        if previous_dataset is not None and dataset != previous_dataset:
            lines.append(r"\midrule")
        dataset_cell = latex(dataset) if dataset != previous_dataset else ""
        setting_cell = latex(setting).replace(r"\_", "--")
        lines.append(
            " & ".join(
                [dataset_cell, setting_cell, rf"$\times 10^{{{exponent}}}$", *cells]
            )
            + r" \\"
        )
        previous_dataset = dataset
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def generate_results_table(
    experiment_dir,
    output=None,
    metric="mse",
    split="test1",
    datasets=None,
    methods=None,
    show_std=False,
    decimals=2,
    settings=None,
):
    root = Path(experiment_dir)
    table = build_table(
        discover(root, split, metric),
        datasets,
        methods,
        split,
        metric,
        show_std,
        decimals,
        settings,
    )
    output = Path(output) if output else root / f"results_{split}_{metric}.tex"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table, encoding="utf-8")
    return output


def _expected_configurations(runs, datasets, settings, methods):
    available = {
        (dataset, setting)
        for dataset, setting, method in runs
        if method in methods
        and (not datasets or dataset in datasets)
        and (not settings or setting in settings)
    }
    if datasets and settings:
        return sorted(product(datasets, settings))
    return sorted(available)


def _format_number(value):
    return "--" if value is None else f"{value:.6g}"


def _summary_tex(summary):
    rows = []
    for method, values in summary["methods"].items():
        rows.append((method, values))
    rows.append(("test-oracle", summary["oracle"]))
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Macro-average {summary['metric'].upper()} on {latex(summary['split'])}. "
        r"Each dataset--setting pair has equal weight. Seed uncertainty is computed "
        r"across seed-level macro averages. The test oracle is an optimistic reference.}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Policy & Mean & Seed std. & Seed variance & Mean rel. gain (\%) \\",
        r"\midrule",
    ]
    for label, values in rows:
        lines.append(
            " & ".join(
                [
                    latex(label),
                    _format_number(values["macro_mean"]),
                    _format_number(values["seed_std"]),
                    _format_number(values["seed_variance"]),
                    _format_number(values["mean_relative_improvement_percent"]),
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def generate_average_summary(
    experiment_dir,
    output,
    metric="mse",
    split="test1",
    datasets=None,
    settings=None,
    methods=None,
    oracle_methods=None,
    baseline_method=None,
    expected_seeds=None,
    strict=False,
):
    """Report equal-configuration macro MSEs and a test-selected oracle.

    The test oracle is deliberately labelled as optimistic: it chooses the lowest
    mean test error for each dataset/setting and is not a deployable selector.
    """
    root = Path(experiment_dir)
    runs = discover_runs(root, split, metric)
    datasets = list(datasets or [])
    settings = [normalise_setting(setting) for setting in settings or []]
    methods = list(methods or sorted({method for _, _, method in runs}))
    oracle_methods = list(oracle_methods or methods)
    unknown_oracle = sorted(set(oracle_methods) - set(methods))
    if unknown_oracle:
        raise ValueError(f"oracle methods are not summary methods: {unknown_oracle}")
    if not methods:
        raise ValueError("at least one summary method is required")
    baseline_method = baseline_method or methods[0]
    if baseline_method not in methods:
        raise ValueError(f"baseline method is not a summary method: {baseline_method}")

    configurations = _expected_configurations(runs, datasets, settings, methods)
    if not configurations:
        raise ValueError("no matching result configurations were found")
    missing = [
        {"dataset": dataset, "setting": setting, "method": method}
        for dataset, setting in configurations
        for method in methods
        if not runs.get((dataset, setting, method))
    ]
    complete = [
        (dataset, setting)
        for dataset, setting in configurations
        if all(runs.get((dataset, setting, method)) for method in methods)
    ]
    if strict and missing:
        raise ValueError(f"missing {len(missing)} dataset/setting/method results")
    if not complete:
        raise ValueError("no complete dataset/setting rows were found")

    seed_sets = [
        set(runs[(dataset, setting, method)])
        for dataset, setting in complete
        for method in methods
    ]
    common_seeds = sorted(set.intersection(*seed_sets))
    expected_seeds = sorted(set(expected_seeds or []))
    missing_seeds = [seed for seed in expected_seeds if seed not in common_seeds]
    if strict and missing_seeds:
        raise ValueError(f"missing complete results for seeds: {missing_seeds}")
    if not common_seeds:
        raise ValueError("no seed is complete across all selected results")

    def cell_mean(dataset, setting, method):
        return fmean(
            runs[(dataset, setting, method)][seed]["mean"] for seed in common_seeds
        )

    baseline_cells = {
        (dataset, setting): cell_mean(dataset, setting, baseline_method)
        for dataset, setting in complete
    }

    def method_summary(method):
        cells = [cell_mean(dataset, setting, method) for dataset, setting in complete]
        seed_macros = [
            fmean(
                runs[(dataset, setting, method)][seed]["mean"]
                for dataset, setting in complete
            )
            for seed in common_seeds
        ]
        within_run_variances = [
            runs[(dataset, setting, method)][seed]["variance"]
            for dataset, setting in complete
            for seed in common_seeds
        ]
        relative = [
            100.0
            * (baseline_cells[(dataset, setting)] - cell_mean(dataset, setting, method))
            / baseline_cells[(dataset, setting)]
            for dataset, setting in complete
            if baseline_cells[(dataset, setting)] != 0
        ]
        return {
            "macro_mean": fmean(cells),
            "seed_std": stdev(seed_macros) if len(seed_macros) > 1 else None,
            "seed_variance": variance(seed_macros) if len(seed_macros) > 1 else None,
            "mean_within_run_loss_variance": fmean(within_run_variances),
            "mean_relative_improvement_percent": fmean(relative) if relative else None,
        }

    method_summaries = {method: method_summary(method) for method in methods}
    selected = {
        (dataset, setting): min(
            oracle_methods, key=lambda method: cell_mean(dataset, setting, method)
        )
        for dataset, setting in complete
    }
    oracle_cells = [
        cell_mean(dataset, setting, selected[(dataset, setting)])
        for dataset, setting in complete
    ]
    oracle_seed_macros = [
        fmean(
            runs[(dataset, setting, selected[(dataset, setting)])][seed]["mean"]
            for dataset, setting in complete
        )
        for seed in common_seeds
    ]
    oracle_variances = [
        runs[(dataset, setting, selected[(dataset, setting)])][seed]["variance"]
        for dataset, setting in complete
        for seed in common_seeds
    ]
    oracle_relative = [
        100.0
        * (baseline_cells[(dataset, setting)] - cell_mean(dataset, setting, method))
        / baseline_cells[(dataset, setting)]
        for (dataset, setting), method in selected.items()
        if baseline_cells[(dataset, setting)] != 0
    ]
    summary = {
        "split": split,
        "metric": metric,
        "aggregation": "equal weight per dataset/setting; paired complete seeds only",
        "configuration_count": len(complete),
        "common_seeds": common_seeds,
        "baseline_method": baseline_method,
        "methods": method_summaries,
        "oracle": {
            "selection": "lowest mean test metric per dataset/setting (optimistic)",
            "candidates": oracle_methods,
            "macro_mean": fmean(oracle_cells),
            "seed_std": stdev(oracle_seed_macros) if len(oracle_seed_macros) > 1 else None,
            "seed_variance": variance(oracle_seed_macros)
            if len(oracle_seed_macros) > 1
            else None,
            "mean_within_run_loss_variance": fmean(oracle_variances),
            "mean_relative_improvement_percent": fmean(oracle_relative)
            if oracle_relative
            else None,
            "selection_counts": dict(sorted(Counter(selected.values()).items())),
            "selections": [
                {"dataset": dataset, "setting": setting, "method": method}
                for (dataset, setting), method in sorted(selected.items())
            ],
        },
        "missing_results": missing,
        "missing_expected_seeds": missing_seeds,
    }

    output = Path(output)
    json_output = output if output.suffix == ".json" else output.with_suffix(".json")
    tex_output = json_output.with_suffix(".tex")
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    tex_output.write_text(_summary_tex(summary), encoding="utf-8")
    return json_output, tex_output


def _csv_arg(value):
    return value.split(",") if value else None


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir")
    parser.add_argument("--output")
    parser.add_argument("--metric", default="mse")
    parser.add_argument("--split", default="test1")
    parser.add_argument("--datasets")
    parser.add_argument("--settings")
    parser.add_argument("--methods")
    parser.add_argument("--show-std", action="store_true")
    parser.add_argument("--decimals", type=int, default=2)
    parser.add_argument("--summary-output")
    parser.add_argument("--summary-methods")
    parser.add_argument("--oracle-methods")
    parser.add_argument("--baseline-method")
    parser.add_argument("--expected-seeds")
    parser.add_argument("--strict-summary", action="store_true")
    args = parser.parse_args()
    datasets = _csv_arg(args.datasets)
    settings = _csv_arg(args.settings)
    methods = _csv_arg(args.methods)
    output = generate_results_table(
        args.experiment_dir,
        args.output,
        args.metric,
        args.split,
        datasets,
        methods,
        args.show_std,
        args.decimals,
        settings,
    )
    LOGGER.info("wrote table path=%s", output)
    if args.summary_output:
        summary_outputs = generate_average_summary(
            args.experiment_dir,
            args.summary_output,
            args.metric,
            args.split,
            datasets,
            settings,
            _csv_arg(args.summary_methods) or methods,
            _csv_arg(args.oracle_methods),
            args.baseline_method,
            [int(seed) for seed in _csv_arg(args.expected_seeds) or []],
            args.strict_summary,
        )
        for summary_output in summary_outputs:
            LOGGER.info("wrote summary path=%s", summary_output)


if __name__ == "__main__":
    main()
