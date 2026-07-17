"""Build seed-aggregated, row-scaled LaTeX tables from RevIN results."""

import argparse
import json
import math
from pathlib import Path
from statistics import fmean, stdev


def discover(root: Path, split: str, metric: str):
    values = {}
    for path in root.rglob("results.json"):
        parts = path.relative_to(root).parts
        if len(parts) < 5:
            continue
        dataset, setting, method = parts[0], parts[1], "/".join(parts[2:-2])
        result = json.loads(path.read_text(encoding="utf-8"))[split][metric]["mean"]
        values.setdefault((dataset, setting, method), []).append(float(result))
    return values


def latex(text):
    return str(text).replace("_", r"\_")


def row_exponent(means):
    finite = sorted(abs(value) for value in means if math.isfinite(value) and value != 0)
    if not finite:
        return 0
    return math.floor(math.log10(finite[len(finite) // 2]))


def build_table(values, datasets, methods, split, metric, show_std, decimals=2):
    rows = sorted({(dataset, setting) for dataset, setting, _ in values})
    rows = [row for row in rows if not datasets or row[0] in datasets]
    methods = methods or sorted({method for _, _, method in values})
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{metric.upper()} on {latex(split)}.}}",
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
            if (dataset, setting, method) in values
        }
        best = min(means.values())
        exponent = row_exponent(means.values())
        divisor = 10.0**exponent
        cells = []
        for method in methods:
            runs = values.get((dataset, setting, method), [])
            if not runs:
                cells.append("--")
                continue
            mean = fmean(runs)
            cell = f"{mean / divisor:.{decimals}f}"
            if show_std and len(runs) > 1:
                cell += rf" $\pm$ {stdev(runs) / divisor:.{decimals}f}"
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
):
    root = Path(experiment_dir)
    table = build_table(
        discover(root, split, metric), datasets, methods, split, metric, show_std, decimals
    )
    output = Path(output) if output else root / f"results_{split}_{metric}.tex"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table, encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir")
    parser.add_argument("--output")
    parser.add_argument("--metric", default="mse")
    parser.add_argument("--split", default="test1")
    parser.add_argument("--datasets")
    parser.add_argument("--methods")
    parser.add_argument("--show-std", action="store_true")
    parser.add_argument("--decimals", type=int, default=2)
    args = parser.parse_args()
    output = generate_results_table(
        args.experiment_dir,
        args.output,
        args.metric,
        args.split,
        args.datasets.split(",") if args.datasets else None,
        args.methods.split(",") if args.methods else None,
        args.show_std,
        args.decimals,
    )
    print(output)


if __name__ == "__main__":
    main()
