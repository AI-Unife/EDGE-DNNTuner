"""
Group-level statistics and an accuracy-vs-HW-cost scatter plot across tuning
experiments, built on top of analyze_results.py.

Two ways to feed it data (combinable):
  --summary-csv PATH [PATH ...]   Already-computed summary CSV(s) from
                                   analyze_results.py (fast path, no recompute).
  --parent-dir DIR [DIR ...]      Parent dir(s): every directory found anywhere
                                   underneath (at any depth -- experiments directly
                                   inside, or grouped under intermediate subfolders)
                                   that contains an algorithm_logs/ subfolder is
                                   analyzed via analyze_results.analyze_experiment().
  --experiment DIR [DIR ...]      Individual experiment dir(s), analyzed the same way.

For --parent-dir/--experiment targets, --hw-weight sets hardware_module's
weight_cost used to (re)compute best_latency/best_hw_cost/best_hw_total_cost
whenever missing -- including experiments whose mod_list never included
hardware_module, so "what would the HW cost have been at weight W" can be
answered for every experiment. Each (experiment, hw_weight) result is cached in
<out-dir>/experiment_stats_cache.csv so re-running with the same weight never
reloads a Keras model twice.

Outputs (under --out-dir, default experiment_stats_out/):
  stats_by_dataset_opt_module.csv   mean/std of accuracy, flops, latency, hw_cost,
                                     hw_total_cost, and the timing_report.csv
                                     aggregates (total/search/symbolic/training
                                     time, plus one column per active module),
                                     grouped by (dataset, opt, module) -- i.e.
                                     across seed repeats of the same configuration.
  scatter_accuracy_vs_hwcost.png    accuracy (y) vs HW total cost (x), colored by
                                     module (no-module / flops-module / hw-module
                                     / flops+hw-module), shaped by dataset, with
                                     one extra larger black-edged marker per
                                     (dataset, module) at the group's mean.

--dataset/--opt/--seed/--module restrict which rows are used, in both outputs
(default: everything found).
"""

import argparse
import ast
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import matplotlib.pyplot as plt

import analyze_results

_CACHE_FILENAME = "experiment_stats_cache.csv"

_METRICS = [
    ("best_accuracy", "accuracy"),
    ("best_flops", "flops"),
    ("best_latency", "latency"),
    ("best_hw_cost", "hw_cost"),
    ("best_hw_total_cost", "hw_total_cost"),
    # From timing_report.csv, aggregated per experiment by get_timing_summary()
    # (see analyze_results.py): mean_* is the mean per-iteration cost, comparable
    # across experiments regardless of how many iterations each ran; total_time_sum
    # is the whole tuning run's wall-clock duration. mean_module_time_<name> columns
    # (one per active module, name varies per experiment) are added dynamically below.
    ("mean_total_time", "total_time"),
    ("total_time_sum", "total_time_sum"),
    ("mean_search_time", "search_time"),
    ("mean_symbolic_time", "symbolic_time"),
    ("mean_training_time", "training_time"),
]

_MODULE_COLORS = {
    "no-module": "#4C72B0",
    "flops-module": "#DD8452",
    "hw-module": "#55A868",
    "flops+hw-module": "#C44E52",
}
_DATASET_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


# ---------------------------------------------------------------------------
# mod_list -> module label
# ---------------------------------------------------------------------------

def _parse_mod_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []
    try:
        parsed = ast.literal_eval(s)
        return parsed if isinstance(parsed, list) else [str(parsed)]
    except Exception:
        return [m.strip().strip("'\"") for m in s.strip("[]").split(",") if m.strip()]


def module_label(mod_list: Any) -> str:
    """"no-module" / "flops-module" / "hw-module" / "flops+hw-module" from a
    config.yaml-style mod_list, whatever form it currently comes in (a real
    list from a fresh analyze_experiment() call, or its string repr from a
    CSV round-trip)."""
    mods = set(_parse_mod_list(mod_list))
    has_flops = "flops_module" in mods
    has_hw = "hardware_module" in mods
    if has_flops and has_hw:
        return "flops+hw-module"
    if has_flops:
        return "flops-module"
    if has_hw:
        return "hw-module"
    return "no-module"


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def discover_experiment_dirs(parent_dirs: Optional[List[str]],
                              experiment_dirs: Optional[List[str]]) -> List[Path]:
    """Every directory with an algorithm_logs/ subfolder found under each
    parent_dir, at ANY depth -- so a parent_dir of plain experiment folders,
    or one holding intermediate subfolders that each group several experiment
    folders (or any deeper nesting), both work the same way. Once a directory
    is recognized as an experiment, its own subtree isn't descended into."""
    dirs = [Path(e).resolve() for e in (experiment_dirs or [])]
    for p in parent_dirs or []:
        p = Path(p).resolve()
        for root, subdirs, _files in os.walk(p):
            root_path = Path(root)
            if (root_path / "algorithm_logs").exists():
                dirs.append(root_path)
                subdirs[:] = []  # an experiment dir doesn't nest another one
                continue
            subdirs.sort()
    seen, out = set(), []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _norm_weight(w: Any) -> Optional[float]:
    if w in (None, ""):
        return None
    return round(float(w), 6)


def _load_cache(cache_path: Path) -> List[Dict[str, Any]]:
    if not cache_path.exists():
        return []
    with open(cache_path, newline="") as f:
        return list(csv.DictReader(f))


def _write_cache(rows: List[Dict[str, Any]], cache_path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(cache_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def collect_rows_from_experiments(exp_dirs: List[Path], hw_weight: float,
                                   cache_path: Path) -> List[Dict[str, Any]]:
    """analyze_results.analyze_experiment() for each dir, reusing a disk cache
    keyed by (experiment, hw_weight) so a repeat run at the same weight never
    reloads a Keras model."""
    existing = _load_cache(cache_path)
    target_weight = _norm_weight(hw_weight)
    by_key = {(r.get("experiment"), _norm_weight(r.get("hw_weight_used"))): r for r in existing}

    all_rows = dict(by_key)  # keep every previously cached (experiment, weight) combo
    rows = []
    for exp_dir in exp_dirs:
        key = (exp_dir.name, target_weight)
        if key in by_key:
            print(f"[cache] {exp_dir.name} (hw_weight={hw_weight}) -- reusing cached result")
            rows.append(by_key[key])
            continue
        row = analyze_results.analyze_experiment(exp_dir, hw_weight=hw_weight)
        if row is not None:
            rows.append(row)
            all_rows[key] = row

    _write_cache(list(all_rows.values()), cache_path)
    return rows


def load_rows_from_summary_csv(paths: List[Path]) -> List[Dict[str, Any]]:
    frames = []
    for p in paths:
        frames.append(pd.read_csv(p, sep=None, engine="python"))
    return pd.concat(frames, ignore_index=True).to_dict("records")


def build_dataframe(args, out_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if args.summary_csv:
        rows += load_rows_from_summary_csv([Path(p) for p in args.summary_csv])

    exp_dirs = discover_experiment_dirs(args.parent_dir, args.experiment)
    if exp_dirs:
        cache_path = out_dir / _CACHE_FILENAME
        rows += collect_rows_from_experiments(exp_dirs, args.hw_weight, cache_path)

    if not rows:
        print("[ERROR] no data: pass --summary-csv and/or --parent-dir / --experiment", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    df["module"] = df["mod_list"].apply(module_label) if "mod_list" in df.columns else "no-module"
    for col, _ in _METRICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Per-module timing columns: name varies per experiment, so not in _METRICS.
    for col in df.columns:
        if col.startswith("mean_module_time_"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "seed" in df.columns:
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    return df


def apply_filters(df: pd.DataFrame, args) -> pd.DataFrame:
    if args.dataset:
        df = df[df["dataset"].isin(args.dataset)]
    if args.opt:
        df = df[df["opt"].isin(args.opt)]
    if args.seed:
        df = df[df["seed"].isin([float(s) for s in args.seed])]
    if args.module:
        df = df[df["module"].isin(args.module)]
    return df


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std of accuracy, flops, latency, hw_cost, hw_total_cost and the
    timing_report.csv aggregates (total/search/symbolic/training time, plus one
    column per active module) per (dataset, opt, module) -- i.e. across whatever
    repeats (seeds) fall in each group. hw_cost is the raw manufacturing-cost
    term; hw_total_cost is the latency-weighted total used for the scatter
    plot's x-axis."""
    group_cols = ["dataset", "opt", "module"]
    module_time_cols = [c for c in df.columns if c.startswith("mean_module_time_")]
    metrics = _METRICS + [(c, c[len("mean_"):]) for c in module_time_cols]

    records = []
    for keys, g in df.groupby(group_cols, dropna=False):
        rec = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        rec["n"] = len(g)
        for col, label in metrics:
            vals = g[col].dropna() if col in g.columns else pd.Series([], dtype=float)
            rec[f"{label}_n"] = int(vals.count())
            rec[f"{label}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            rec[f"{label}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
        records.append(rec)
    return pd.DataFrame(records).sort_values(group_cols).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Scatter plot
# ---------------------------------------------------------------------------

def plot_scatter(df: pd.DataFrame, out_path: Path) -> None:
    plot_df = df.dropna(subset=["best_hw_total_cost", "best_accuracy"])
    if plot_df.empty:
        print("[WARNING] no rows have both best_hw_total_cost and best_accuracy -- "
              "scatter plot skipped. Pass --hw-weight together with --parent-dir/--experiment "
              "to fill in missing HW cost for experiments that didn't use hardware_module.")
        return

    datasets = sorted(plot_df["dataset"].dropna().unique())
    markers = {ds: _DATASET_MARKERS[i % len(_DATASET_MARKERS)] for i, ds in enumerate(datasets)}

    fig, ax = plt.subplots(figsize=(9, 7))
    for (ds, mod), g in plot_df.groupby(["dataset", "module"]):
        color = _MODULE_COLORS.get(mod, "#888888")
        marker = markers.get(ds, "o")
        ax.scatter(g["best_hw_total_cost"], g["best_accuracy"] * 100,
                   c=color, marker=marker, alpha=0.65, s=70,
                   edgecolors="white", linewidths=0.5)
        ax.scatter([g["best_hw_total_cost"].mean()], [g["best_accuracy"].mean() * 100],
                   c=color, marker=marker, s=260, edgecolors="black", linewidths=1.6, zorder=5)

    module_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=m)
        for m, c in _MODULE_COLORS.items() if m in plot_df["module"].unique()
    ]
    dataset_handles = [
        plt.Line2D([0], [0], marker=markers[ds], color="gray", linestyle="", markersize=9, label=ds)
        for ds in datasets
    ]
    mean_handle = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                              markeredgecolor="black", markeredgewidth=1.6, markersize=11,
                              label="mean per dataset x module")

    leg1 = ax.legend(handles=module_handles, title="module (color)", loc="upper left",
                      bbox_to_anchor=(1.02, 1.0), frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=dataset_handles + [mean_handle], title="dataset (shape)", loc="upper left",
              bbox_to_anchor=(1.02, 0.55), frameon=False)

    ax.set_xlabel("HW total cost")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs HW total cost")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Scatter plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary-csv", type=str, nargs="+", default=None,
                   help="Already-computed summary CSV(s) from analyze_results.py (fast path, no recompute).")
    p.add_argument("--parent-dir", type=str, nargs="+", default=None,
                   help="Parent dir(s): every directory found anywhere underneath (at any depth) with an "
                        "algorithm_logs/ subfolder is analyzed as an experiment -- works whether experiments "
                        "sit directly inside or are grouped under intermediate subfolders.")
    p.add_argument("--experiment", type=str, nargs="+", default=None,
                   help="Individual experiment dir(s) to analyze directly.")
    p.add_argument("--hw-weight", type=float, default=0.3,
                   help="hardware_module weight_cost used to (re)compute HW cost for --parent-dir/--experiment "
                        "targets whenever it's missing -- including experiments that never used hardware_module. "
                        "Default: 0.3. Cached per (experiment, hw_weight) in <out-dir>/experiment_stats_cache.csv.")
    p.add_argument("--dataset", type=str, nargs="+", default=None, help="Restrict to these dataset(s). Default: all.")
    p.add_argument("--opt", type=str, nargs="+", default=None, help="Restrict to these opt(s). Default: all.")
    p.add_argument("--seed", type=str, nargs="+", default=None, help="Restrict to these seed(s). Default: all.")
    p.add_argument("--module", type=str, nargs="+", default=None,
                   choices=["no-module", "flops-module", "hw-module", "flops+hw-module"],
                   help="Restrict to these module configuration(s). Default: all.")
    p.add_argument("--out-dir", type=str, default="experiment_stats_out")
    args = p.parse_args()

    if not args.summary_csv and not args.parent_dir and not args.experiment:
        p.error("give --summary-csv and/or --parent-dir / --experiment")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataframe(args, out_dir)
    df = apply_filters(df, args)
    if df.empty:
        print("[ERROR] no rows left after filtering.", file=sys.stderr)
        sys.exit(1)

    print(f"\n[plan] {len(df)} experiment(s) after filtering -- "
          f"datasets={sorted(df['dataset'].dropna().unique())}, "
          f"opts={sorted(df['opt'].dropna().unique())}, "
          f"modules={sorted(df['module'].dropna().unique())}")

    stats = compute_stats(df)
    stats_path = out_dir / "stats_by_dataset_opt_module.csv"
    stats.to_csv(stats_path, index=False)
    print(f"\nStats saved: {stats_path}\n")
    print(stats.to_string(index=False))

    plot_scatter(df, out_dir / "scatter_accuracy_vs_hwcost.png")


if __name__ == "__main__":
    main()
