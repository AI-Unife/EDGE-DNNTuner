#!/usr/bin/env python3
"""Export downloaded BANANAS/FlexiBO summaries to an ordered CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev
from typing import Any


DEFAULT_BANANAS_DIR = Path("results_BANANAS_naszilla_controller_cluster")
DEFAULT_FLEXIBO_DIR = Path("results_FLEXIBO_controller_cluster")
DEFAULT_OUTPUT = Path("tuner_results_summary.csv")
DEFAULT_AGGREGATES_OUTPUT = Path("tuner_results_aggregates.csv")
DEFAULT_PAPER_TABLE_OUTPUT = Path("tuner_results_paper_table.csv")


FIELDNAMES = [
    "tuner",
    "dataset",
    "seed",
    "experiment",
    "progress_evals",
    "events",
    "target_evals",
    "progress_fraction",
    "best_score",
    "best_score_event",
    "best_score_accuracy",
    "best_score_flops",
    "best_score_params",
    "best_accuracy",
    "best_accuracy_event",
    "best_accuracy_flops",
    "best_accuracy_params",
    "invalid_or_penalty_events",
    "history_path",
]

AGGREGATE_FIELDNAMES = [
    "group_type",
    "tuner",
    "dataset",
    "seed",
    "runs",
    "mean_progress_evals",
    "mean_progress_fraction",
    "mean_best_score",
    "std_best_score",
    "min_best_score",
    "max_best_score",
    "mean_best_score_event",
    "std_best_score_event",
    "mean_best_score_accuracy",
    "std_best_score_accuracy",
    "mean_best_accuracy",
    "std_best_accuracy",
    "min_best_accuracy",
    "max_best_accuracy",
    "mean_best_score_flops",
    "mean_best_score_params",
    "mean_best_accuracy_flops",
    "mean_best_accuracy_params",
    "mean_invalid_or_penalty_events",
]

PAPER_TABLE_FIELDNAMES = [
    "dataset",
    "strategy",
    "runs",
    "Best Score",
    "Accuracy",
    "MFLOPs",
    "N. Iteration",
    "best_score_rank",
    "accuracy_rank",
    "mflops_rank",
    "n_iteration_rank",
    "best_score_mean",
    "best_score_std",
    "accuracy_pct_mean",
    "accuracy_pct_std",
    "mflops_mean",
    "mflops_std",
    "n_iteration_mean",
    "n_iteration_std",
]


def float_or_none(value: Any) -> float | None:
    if value in (None, "", "None", "nan", "NaN"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def int_or_blank(value: float | int | None) -> int | str:
    if value is None:
        return ""
    return int(round(float(value)))


def float_or_blank(value: float | None) -> float | str:
    if value is None:
        return ""
    return value


def format_mean_std(value: float | str, deviation: float | str, digits: int = 3, suffix: str = "") -> str:
    if value == "" or deviation == "":
        return ""
    return f"{float(value):.{digits}f}{suffix} (+- {float(deviation):.{digits}f})"


def parse_target_evals(experiment: str) -> int | str:
    for part in experiment.split("_"):
        if part.startswith("e") and part[1:].isdigit():
            return int(part[1:])
    return ""


def parse_experiment_name(experiment: str) -> tuple[str, str]:
    parts = experiment.split("_")
    if len(parts) < 4:
        return "", ""
    return parts[1], parts[2].replace("seed", "")


def read_optional_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(errors="replace").splitlines()


def read_accuracy_report(path: Path) -> list[float | None]:
    values: list[float | None] = []
    for line in read_optional_lines(path):
        text = line.strip()
        values.append(None if text.startswith("None") or not text else float_or_none(text))
    return values


def read_flops_report(path: Path) -> tuple[list[float | None], list[float | None]]:
    params: list[float | None] = []
    flops: list[float | None] = []
    for line in read_optional_lines(path):
        parts = line.split()
        if len(parts) >= 2:
            params.append(float_or_none(parts[0]))
            flops.append(float_or_none(parts[1]))
        else:
            params.append(None)
            flops.append(None)
    return params, flops


def get_indexed(values: list[Any], index: int | None) -> Any:
    if index is None or index < 0 or index >= len(values):
        return None
    return values[index]


def base_row(tuner: str, experiment: str, history_path: Path, progress: int, events: int) -> dict[str, Any]:
    dataset, seed = parse_experiment_name(experiment)
    target = parse_target_evals(experiment)
    fraction = progress / target if isinstance(target, int) and target > 0 else ""
    return {
        "tuner": tuner,
        "dataset": dataset,
        "seed": seed,
        "experiment": experiment,
        "progress_evals": progress,
        "events": events,
        "target_evals": target,
        "progress_fraction": fraction,
        "history_path": str(history_path),
    }


def parse_bananas(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for history_path in sorted(root.glob("*/algorithm_logs/bananas_history.csv")):
        experiment = history_path.parents[1].name
        log_dir = history_path.parent
        history = list(csv.DictReader(history_path.open(newline="")))
        if not history:
            continue

        scores = [float_or_none(row.get("score")) for row in history]
        valid_score_indices = [i for i, score in enumerate(scores) if score is not None]
        if not valid_score_indices:
            continue

        accuracies = read_accuracy_report(log_dir / "acc_report.txt")
        params, flops = read_flops_report(log_dir / "flops_report.txt")

        best_score_i = min(valid_score_indices, key=lambda i: scores[i] or float("inf"))
        valid_acc_indices = [i for i, acc in enumerate(accuracies) if acc is not None]
        best_acc_i = max(valid_acc_indices, key=lambda i: accuracies[i] or float("-inf")) if valid_acc_indices else None

        row = base_row("BANANAS", experiment, history_path, progress=len(history), events=len(history))
        row.update(
            {
                "best_score": float_or_blank(scores[best_score_i]),
                "best_score_event": best_score_i + 1,
                "best_score_accuracy": float_or_blank(get_indexed(accuracies, best_score_i)),
                "best_score_flops": int_or_blank(get_indexed(flops, best_score_i)),
                "best_score_params": int_or_blank(get_indexed(params, best_score_i)),
                "best_accuracy": float_or_blank(get_indexed(accuracies, best_acc_i)),
                "best_accuracy_event": "" if best_acc_i is None else best_acc_i + 1,
                "best_accuracy_flops": int_or_blank(get_indexed(flops, best_acc_i)),
                "best_accuracy_params": int_or_blank(get_indexed(params, best_acc_i)),
                "invalid_or_penalty_events": sum(1 for score in scores if score is not None and score >= 1e9),
            }
        )
        rows.append(row)
    return rows


def parse_flexibo(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for history_path in sorted(root.glob("*/algorithm_logs/flexibo_history.csv")):
        experiment = history_path.parents[1].name
        history = list(csv.DictReader(history_path.open(newline="")))
        if not history:
            continue

        scored = [row for row in history if float_or_none(row.get("score")) is not None]
        if not scored:
            continue

        best_score_row = min(scored, key=lambda row: float_or_none(row.get("score")) or float("inf"))
        accuracy_rows = [row for row in scored if float_or_none(row.get("accuracy")) is not None]
        best_acc_row = (
            max(accuracy_rows, key=lambda row: float_or_none(row.get("accuracy")) or float("-inf"))
            if accuracy_rows
            else None
        )
        progress = sum(1 for row in history if row.get("objective") in ("accuracy", "both"))

        row = base_row("FlexiBO", experiment, history_path, progress=progress, events=len(history))
        row.update(
            {
                "best_score": float_or_blank(float_or_none(best_score_row.get("score"))),
                "best_score_event": int_or_blank(float_or_none(best_score_row.get("event"))),
                "best_score_accuracy": float_or_blank(float_or_none(best_score_row.get("accuracy"))),
                "best_score_flops": int_or_blank(float_or_none(best_score_row.get("flops"))),
                "best_score_params": int_or_blank(float_or_none(best_score_row.get("params"))),
                "best_accuracy": float_or_blank(float_or_none(best_acc_row.get("accuracy")) if best_acc_row else None),
                "best_accuracy_event": int_or_blank(float_or_none(best_acc_row.get("event")) if best_acc_row else None),
                "best_accuracy_flops": int_or_blank(float_or_none(best_acc_row.get("flops")) if best_acc_row else None),
                "best_accuracy_params": int_or_blank(float_or_none(best_acc_row.get("params")) if best_acc_row else None),
                "invalid_or_penalty_events": sum(
                    1 for row in scored if (float_or_none(row.get("score")) or 0) >= 1e9
                ),
            }
        )
        rows.append(row)
    return rows


def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    dataset_order = {"cifar10": 0, "cifar100": 1}
    tuner_order = {"BANANAS": 0, "FlexiBO": 1}
    return (
        dataset_order.get(row["dataset"], 99),
        tuner_order.get(row["tuner"], 99),
        int(row["seed"]) if str(row["seed"]).isdigit() else 999999,
    )


def numeric_values(rows: list[dict[str, Any]], column: str) -> list[float]:
    return [float(row[column]) for row in rows if row.get(column) not in ("", None)]


def aggregate_value(rows: list[dict[str, Any]], column: str, op: str) -> float | str:
    values = numeric_values(rows, column)
    if not values:
        return ""
    if op == "mean":
        return mean(values)
    if op == "std":
        return stdev(values) if len(values) > 1 else 0.0
    if op == "min":
        return min(values)
    if op == "max":
        return max(values)
    raise ValueError(f"unknown aggregate op: {op}")


def aggregate_row(group_type: str, rows: list[dict[str, Any]], tuner: str = "", dataset: str = "", seed: str = "") -> dict[str, Any]:
    return {
        "group_type": group_type,
        "tuner": tuner,
        "dataset": dataset,
        "seed": seed,
        "runs": len(rows),
        "mean_progress_evals": aggregate_value(rows, "progress_evals", "mean"),
        "mean_progress_fraction": aggregate_value(rows, "progress_fraction", "mean"),
        "mean_best_score": aggregate_value(rows, "best_score", "mean"),
        "std_best_score": aggregate_value(rows, "best_score", "std"),
        "min_best_score": aggregate_value(rows, "best_score", "min"),
        "max_best_score": aggregate_value(rows, "best_score", "max"),
        "mean_best_score_event": aggregate_value(rows, "best_score_event", "mean"),
        "std_best_score_event": aggregate_value(rows, "best_score_event", "std"),
        "mean_best_score_accuracy": aggregate_value(rows, "best_score_accuracy", "mean"),
        "std_best_score_accuracy": aggregate_value(rows, "best_score_accuracy", "std"),
        "mean_best_accuracy": aggregate_value(rows, "best_accuracy", "mean"),
        "std_best_accuracy": aggregate_value(rows, "best_accuracy", "std"),
        "min_best_accuracy": aggregate_value(rows, "best_accuracy", "min"),
        "max_best_accuracy": aggregate_value(rows, "best_accuracy", "max"),
        "mean_best_score_flops": aggregate_value(rows, "best_score_flops", "mean"),
        "mean_best_score_params": aggregate_value(rows, "best_score_params", "mean"),
        "mean_best_accuracy_flops": aggregate_value(rows, "best_accuracy_flops", "mean"),
        "mean_best_accuracy_params": aggregate_value(rows, "best_accuracy_params", "mean"),
        "mean_invalid_or_penalty_events": aggregate_value(rows, "invalid_or_penalty_events", "mean"),
    }


def build_aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []

    tuners = sorted({row["tuner"] for row in rows})
    datasets = sorted({row["dataset"] for row in rows})
    seeds = sorted({row["seed"] for row in rows}, key=lambda seed: int(seed) if str(seed).isdigit() else 999999)

    for tuner in tuners:
        tuner_rows = [row for row in rows if row["tuner"] == tuner]
        aggregates.append(aggregate_row("tuner", tuner_rows, tuner=tuner))

    for tuner in tuners:
        for dataset in datasets:
            group = [row for row in rows if row["tuner"] == tuner and row["dataset"] == dataset]
            if group:
                aggregates.append(aggregate_row("tuner_dataset_mean_over_seeds", group, tuner=tuner, dataset=dataset))

    for tuner in tuners:
        for seed in seeds:
            group = [row for row in rows if row["tuner"] == tuner and row["seed"] == seed]
            if group:
                aggregates.append(aggregate_row("tuner_seed_mean_over_datasets", group, tuner=tuner, seed=seed))

    return aggregates


def build_paper_table_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paper_rows: list[dict[str, Any]] = []
    datasets = sorted({row["dataset"] for row in rows}, key=lambda dataset: {"cifar10": 0, "cifar100": 1}.get(dataset, 99))
    tuners = sorted({row["tuner"] for row in rows}, key=lambda tuner: {"BANANAS": 0, "FlexiBO": 1}.get(tuner, 99))

    for dataset in datasets:
        for tuner in tuners:
            group = [row for row in rows if row["dataset"] == dataset and row["tuner"] == tuner]
            if not group:
                continue
            score_mean = aggregate_value(group, "best_score", "mean")
            score_std = aggregate_value(group, "best_score", "std")
            accuracy_mean = aggregate_value(group, "best_score_accuracy", "mean")
            accuracy_std = aggregate_value(group, "best_score_accuracy", "std")
            flops_mean = aggregate_value(group, "best_score_flops", "mean")
            flops_std = aggregate_value(group, "best_score_flops", "std")
            iteration_mean = aggregate_value(group, "best_score_event", "mean")
            iteration_std = aggregate_value(group, "best_score_event", "std")

            accuracy_pct_mean = "" if accuracy_mean == "" else float(accuracy_mean) * 100.0
            accuracy_pct_std = "" if accuracy_std == "" else float(accuracy_std) * 100.0
            mflops_mean = "" if flops_mean == "" else float(flops_mean) / 1_000_000.0
            mflops_std = "" if flops_std == "" else float(flops_std) / 1_000_000.0

            paper_rows.append(
                {
                    "dataset": dataset,
                    "strategy": tuner,
                    "runs": len(group),
                    "Best Score": format_mean_std(score_mean, score_std, digits=3),
                    "Accuracy": format_mean_std(accuracy_pct_mean, accuracy_pct_std, digits=2, suffix="%"),
                    "MFLOPs": format_mean_std(mflops_mean, mflops_std, digits=1),
                    "N. Iteration": format_mean_std(iteration_mean, iteration_std, digits=1),
                    "best_score_mean": score_mean,
                    "best_score_std": score_std,
                    "accuracy_pct_mean": accuracy_pct_mean,
                    "accuracy_pct_std": accuracy_pct_std,
                    "mflops_mean": mflops_mean,
                    "mflops_std": mflops_std,
                    "n_iteration_mean": iteration_mean,
                    "n_iteration_std": iteration_std,
                }
            )

    for dataset in datasets:
        dataset_rows = [row for row in paper_rows if row["dataset"] == dataset]
        rank_specs = [
            ("best_score_mean", "best_score_rank", False),
            ("accuracy_pct_mean", "accuracy_rank", True),
            ("mflops_mean", "mflops_rank", False),
            ("n_iteration_mean", "n_iteration_rank", False),
        ]
        for value_key, rank_key, higher_is_better in rank_specs:
            ranked = [
                row
                for row in dataset_rows
                if row.get(value_key) not in ("", None)
            ]
            ranked.sort(key=lambda row: float(row[value_key]), reverse=higher_is_better)
            for rank, row in enumerate(ranked, start=1):
                row[rank_key] = rank

    return paper_rows


def print_console_summary(rows: list[dict[str, Any]]) -> None:
    print(f"Wrote {len(rows)} run summaries.")
    for tuner in ("BANANAS", "FlexiBO"):
        for dataset in ("cifar10", "cifar100"):
            group = [row for row in rows if row["tuner"] == tuner and row["dataset"] == dataset]
            if not group:
                continue
            progress = ", ".join(f"seed{row['seed']}={row['progress_evals']}" for row in group)
            scores = [float(row["best_score"]) for row in group if row["best_score"] != ""]
            accs = [float(row["best_accuracy"]) for row in group if row["best_accuracy"] != ""]
            print(
                f"{tuner} {dataset}: runs={len(group)}, "
                f"mean_best_score={mean(scores):.4f}, "
                f"mean_best_accuracy={mean(accs):.4f}, "
                f"progress=[{progress}]"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bananas-dir", type=Path, default=DEFAULT_BANANAS_DIR)
    parser.add_argument("--flexibo-dir", type=Path, default=DEFAULT_FLEXIBO_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--aggregates-output", type=Path, default=DEFAULT_AGGREGATES_OUTPUT)
    parser.add_argument("--paper-table-output", type=Path, default=DEFAULT_PAPER_TABLE_OUTPUT)
    args = parser.parse_args()

    rows = parse_bananas(args.bananas_dir) + parse_flexibo(args.flexibo_dir)
    rows.sort(key=sort_key)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    aggregate_rows = build_aggregate_rows(rows)
    args.aggregates_output.parent.mkdir(parents=True, exist_ok=True)
    with args.aggregates_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AGGREGATE_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(aggregate_rows)

    paper_table_rows = build_paper_table_rows(rows)
    args.paper_table_output.parent.mkdir(parents=True, exist_ok=True)
    with args.paper_table_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAPER_TABLE_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(paper_table_rows)

    print(f"Output: {args.output}")
    print(f"Aggregates output: {args.aggregates_output}")
    print(f"Paper table output: {args.paper_table_output}")
    print_console_summary(rows)


if __name__ == "__main__":
    main()
