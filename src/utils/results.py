"""Build seed-aggregated tables and macro summaries from RevIN results."""

import argparse
import json
import logging
import math
import re
import sys
from collections import Counter
from itertools import product
from pathlib import Path
from statistics import fmean, stdev, variance


LOGGER = logging.getLogger(__name__)


def normalise_setting(setting):
    """Convert CLI forms such as ``168:24`` and ``168-24`` to ``168_24``."""
    return str(setting).replace(":", "_").replace("-", "_")


def _history_validation_mean(path: Path, split: str, metric: str):
    """Read a final validation mean from an older run's history artifact."""
    history_path = path.with_name("history.pt")
    if not history_path.exists():
        return None
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            f"{path} has no {split} metrics and reading {history_path} requires torch"
        ) from error
    history = torch.load(history_path, map_location="cpu", weights_only=False)
    records = history.get("valid", {}).get(split, [])
    if not records:
        return None
    losses = records[-1].get("losses", {})
    return None if metric not in losses else float(losses[metric])


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
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get(split, {}).get(metric)
        if summary is None and split.startswith("valid"):
            mean = _history_validation_mean(path, split, metric)
            if mean is None:
                continue
            summary = {"mean": mean}
        if summary is None:
            continue
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


def validation_split_for(split):
    match = re.fullmatch(r"test(.*)", str(split))
    if match is None:
        raise ValueError(f"cannot infer validation split corresponding to {split}")
    return f"valid{match.group(1)}"


def _common_cell_seeds(runs, dataset, setting, methods):
    seed_sets = [
        set(runs.get((dataset, setting, method), {})) for method in methods
    ]
    return sorted(set.intersection(*seed_sets)) if seed_sets else []


def _selected_policy_values(
    test_runs,
    selection_runs,
    datasets,
    settings,
    candidate_methods,
):
    """Select one candidate per task and return its seed-level test values."""
    selected_values = {}
    selected_methods = {}
    rows = sorted({(dataset, setting) for dataset, setting, _ in test_runs})
    for dataset, setting in rows:
        if datasets and dataset not in datasets:
            continue
        if settings and setting not in settings:
            continue
        test_seeds = _common_cell_seeds(
            test_runs, dataset, setting, candidate_methods
        )
        selection_seeds = _common_cell_seeds(
            selection_runs, dataset, setting, candidate_methods
        )
        common_seeds = sorted(set(test_seeds) & set(selection_seeds))
        if not common_seeds:
            continue
        selected = min(
            candidate_methods,
            key=lambda method: fmean(
                selection_runs[(dataset, setting, method)][seed]["mean"]
                for seed in common_seeds
            ),
        )
        selected_methods[(dataset, setting)] = selected
        selected_values[(dataset, setting)] = [
            test_runs[(dataset, setting, selected)][seed]["mean"]
            for seed in common_seeds
        ]
    return selected_values, selected_methods


def build_table(
    values,
    datasets,
    methods,
    split,
    metric,
    show_std,
    decimals=2,
    settings=None,
    policy_values=None,
):
    settings = {normalise_setting(setting) for setting in settings or []}
    rows = sorted({(dataset, setting) for dataset, setting, _ in values})
    rows = [
        row
        for row in rows
        if (not datasets or row[0] in datasets) and (not settings or row[1] in settings)
    ]
    methods = methods or sorted({method for _, _, method in values})
    policy_values = policy_values or {}
    policy_labels = list(policy_values)
    uncertainty = " Seed uncertainty is the sample standard deviation." if show_std else ""
    column_spec = f"llc{'r' * len(methods)}"
    if policy_labels:
        column_spec += f"{'r' * (len(policy_labels) - 1)}|r"
    policy_note = (
        " Validation-selected uses the corresponding validation split; "
        "the test oracle is optimistic."
        if policy_labels
        else ""
    )
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{metric.upper()} on {latex(split)}.{uncertainty}{policy_note}}}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        "Dataset & $L$--$H$ & Scale & "
        + " & ".join(map(latex, [*methods, *policy_labels]))
        + r" \\",
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
        for label in policy_labels:
            seed_values = policy_values[label].get((dataset, setting), [])
            if not seed_values:
                cells.append("--")
                continue
            mean = fmean(seed_values)
            cell = f"{mean / divisor:.{decimals}f}"
            if show_std and len(seed_values) > 1:
                cell += rf" $\pm$ {stdev(seed_values) / divisor:.{decimals}f}"
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
    selection_methods=None,
):
    root = Path(experiment_dir)
    test_runs = discover_runs(root, split, metric)
    settings_filter = {
        normalise_setting(setting) for setting in settings or []
    }
    policy_values = {}
    if selection_methods:
        validation_runs = discover_runs(root, validation_split_for(split), metric)
        valid_values, _ = _selected_policy_values(
            test_runs,
            validation_runs,
            datasets,
            settings_filter,
            selection_methods,
        )
        oracle_values, _ = _selected_policy_values(
            test_runs,
            test_runs,
            datasets,
            settings_filter,
            selection_methods,
        )
        policy_values = {
            "validation-selected": valid_values,
            "test-oracle": oracle_values,
        }
    table = build_table(
        {
            key: [record["mean"] for record in seed_records.values()]
            for key, seed_records in test_runs.items()
        },
        datasets,
        methods,
        split,
        metric,
        show_std,
        decimals,
        settings,
        policy_values,
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
    rows.append(("validation-selected", summary["validation_selected"]))
    rows.append((None, None))
    rows.append(("test-oracle", summary["oracle"]))
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Macro-average {summary['metric'].upper()} on {latex(summary['split'])}. "
        r"Each dataset--setting pair has equal weight. Seed uncertainty is computed "
        r"across seed-level macro averages. Validation-selected chooses a method "
        r"on the corresponding validation split; the test oracle is optimistic.}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Policy & Mean & Seed std. & Seed variance & Mean rel. gain (\%) \\",
        r"\midrule",
    ]
    for label, values in rows:
        if label is None:
            lines.append(r"\midrule")
            continue
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
    """Report fixed methods, validation selection, and a test-selected oracle.

    Selection uses the corresponding validation split and the test oracle is
    deliberately optimistic. Both choose one method per dataset/setting using
    the mean metric across the same complete seeds.
    """
    root = Path(experiment_dir)
    runs = discover_runs(root, split, metric)
    validation_split = validation_split_for(split)
    validation_runs = discover_runs(root, validation_split, metric)
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

    missing_validation = [
        {"dataset": dataset, "setting": setting, "method": method}
        for dataset, setting in complete
        for method in oracle_methods
        if not validation_runs.get((dataset, setting, method))
    ]
    if missing_validation:
        raise ValueError(
            f"missing {len(missing_validation)} validation results for "
            f"{validation_split}; rerun affected experiments or retain history.pt"
        )
    validation_seed_sets = [
        set(validation_runs[(dataset, setting, method)])
        for dataset, setting in complete
        for method in oracle_methods
    ]
    validation_common_seeds = sorted(set.intersection(*validation_seed_sets))
    if validation_common_seeds != common_seeds:
        missing_for_selection = sorted(set(common_seeds) - set(validation_common_seeds))
        if strict and missing_for_selection:
            raise ValueError(
                f"missing complete {validation_split} results for seeds: "
                f"{missing_for_selection}"
            )
        common_seeds = sorted(set(common_seeds) & set(validation_common_seeds))
    if not common_seeds:
        raise ValueError("no seed has complete validation and test results")

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
    validation_selected = {
        (dataset, setting): min(
            oracle_methods,
            key=lambda method: fmean(
                validation_runs[(dataset, setting, method)][seed]["mean"]
                for seed in common_seeds
            ),
        )
        for dataset, setting in complete
    }
    selected = {
        (dataset, setting): min(
            oracle_methods, key=lambda method: cell_mean(dataset, setting, method)
        )
        for dataset, setting in complete
    }

    def policy_summary(policy_selection, selection):
        cells = [
            cell_mean(dataset, setting, policy_selection[(dataset, setting)])
            for dataset, setting in complete
        ]
        seed_macros = [
            fmean(
                runs[
                    (dataset, setting, policy_selection[(dataset, setting)])
                ][seed]["mean"]
                for dataset, setting in complete
            )
            for seed in common_seeds
        ]
        within_run_variances = [
            runs[
                (dataset, setting, policy_selection[(dataset, setting)])
            ][seed]["variance"]
            for dataset, setting in complete
            for seed in common_seeds
        ]
        relative = [
            100.0
            * (
                baseline_cells[(dataset, setting)]
                - cell_mean(dataset, setting, method)
            )
            / baseline_cells[(dataset, setting)]
            for (dataset, setting), method in policy_selection.items()
            if baseline_cells[(dataset, setting)] != 0
        ]
        return {
            "selection": selection,
            "candidates": oracle_methods,
            "macro_mean": fmean(cells),
            "seed_std": stdev(seed_macros) if len(seed_macros) > 1 else None,
            "seed_variance": variance(seed_macros) if len(seed_macros) > 1 else None,
            "mean_within_run_loss_variance": fmean(within_run_variances),
            "mean_relative_improvement_percent": fmean(relative) if relative else None,
            "selection_counts": dict(
                sorted(Counter(policy_selection.values()).items())
            ),
            "selections": [
                {"dataset": dataset, "setting": setting, "method": method}
                for (dataset, setting), method in sorted(policy_selection.items())
            ],
        }

    summary = {
        "split": split,
        "metric": metric,
        "aggregation": "equal weight per dataset/setting; paired complete seeds only",
        "configuration_count": len(complete),
        "common_seeds": common_seeds,
        "baseline_method": baseline_method,
        "methods": method_summaries,
        "validation_selected": policy_summary(
            validation_selected,
            f"lowest mean {validation_split} metric per dataset/setting",
        ),
        "oracle": policy_summary(
            selected, "lowest mean test metric per dataset/setting (optimistic)"
        ),
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
        stream=sys.stdout,
        force=True,
    )
    logging.captureWarnings(True)
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
    parser.add_argument("--selection-methods")
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
        _csv_arg(args.selection_methods) or _csv_arg(args.oracle_methods),
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
