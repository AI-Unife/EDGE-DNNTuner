"""
Per-experiment efficiency metrics -- total wall-clock time, memory-bandwidth
latency, and discard ratio -- with the usual dataset/opt/module/w_HW grouped
mean/std (e.g. distinguishing no-module, flops-module, and hardware_module at
each w_HW actually used, such as HW w=0.3 vs HW w=0.7).

Everything here comes straight from algorithm_logs/ and the .out file (no
TensorFlow / model loading, unlike analyze_results.py's FLOPS/HW recompute
fallbacks), so it's fast even over many experiments.

  --parent-dir DIR [DIR ...]   Parent dir(s): every directory found anywhere
                                underneath (at any depth -- experiments directly
                                inside, or grouped under intermediate subfolders)
                                with an algorithm_logs/ subfolder is analyzed.
  --experiment DIR [DIR ...]   Individual experiment dir(s), analyzed the same way.

Per experiment:
  total_iterations       Number of logged iterations (acc_report.txt lines),
                         trained and discarded alike.
  total_time_s          Whole tuning run's wall-clock duration, parsed from the
                         .out file's "TOTAL TIME --------> X seconds" line
                         (printed by symbolic_tuner.py's main() at the end of
                         the run). When that line is missing -- the run never
                         printed it, i.e. it hit the job's wall-clock limit
                         instead of converging -- this is filled in with
                         --timeout-hours (default 24h = 86400s) rather than
                         left blank.
  bandwidth_latency_38_4gbps_s / bandwidth_latency_25_6gbps_s
                         size of the best model in KB, divided by a fixed memory
                         bandwidth of 38.4 GB/s or 25.6 GB/s respectively (1 GB/s
                         = 1e6 KB/s). The size is the on-disk size of
                         <experiment>/Model/best-model.keras when that file was
                         actually saved; otherwise it falls back to the KB TF's
                         own model.summary() reported for the best iteration in
                         the .out file ("Total params: N (X KB/MB)"). Which one
                         was used is recorded per experiment in
                         model_size_source ("file" / "tf_summary" / None).
  discard_ratio          Fraction of logged iterations whose acc_report.txt line
                         was "None" -- i.e. the sampled network violated a
                         constraint (see controller.training()) and was
                         discarded before ever being trained.

Outputs (under --out-dir, default experiment_efficiency_out/):
  experiment_efficiency.csv                 one row per experiment: all fields
                                             above plus dataset/opt/module/w_HW/seed.
  efficiency_stats_by_dataset_opt_wHW.csv   mean/std/n of the four metrics
                                             above, grouped by (dataset, opt,
                                             module, w_HW) -- i.e. across seed
                                             repeats of the same configuration.
                                             w_HW is blanked out for modules
                                             other than hardware_module, since
                                             it was never actually applied then.

--dataset/--opt/--seed restrict which rows are used, in both outputs (default:
everything found).
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import analyze_results as ar
from experiment_stats import discover_experiment_dirs, module_label

_BANDWIDTHS_GBPS = [38.4, 25.6]
_DEFAULT_TIMEOUT_HOURS = 24.0

_METRICS = [
    ("total_iterations", "total_iterations"),
    ("total_time_s", "total_time"),
    ("bandwidth_latency_38_4gbps_s", "bandwidth_latency_38_4gbps"),
    ("bandwidth_latency_25_6gbps_s", "bandwidth_latency_25_6gbps"),
    ("discard_ratio", "discard_ratio"),
]


def best_model_size_kb(exp_dir: Path, analyzer: ar.ResultsAnalyzer,
                        best_iteration: Optional[int]) -> Tuple[Optional[float], Optional[str]]:
    """(size_kb, source) for the best model of this experiment.

    Prefers the size on disk of <exp_dir>/Model/best-model.keras. When that
    file wasn't actually saved (e.g. the run never improved on its initial
    score), falls back to the KB TF's own model.summary() reported for the
    best iteration in the .out file ('Total params: N (X KB/MB)') -- source
    is "file" or "tf_summary" respectively, or (None, None) if neither is
    available."""
    model_path = exp_dir / "Model" / "best-model.keras"
    if model_path.exists():
        return model_path.stat().st_size / 1024.0, "file"
    if best_iteration is not None:
        size_kb = analyzer.get_model_size_kb_from_summary(best_iteration)
        if size_kb is not None:
            print(f"  best-model.keras not saved -- using TF summary size for "
                  f"iteration {best_iteration}: {size_kb:.2f} KB")
            return size_kb, "tf_summary"
    return None, None


def bandwidth_latency_seconds(size_kb: Optional[float], bandwidth_gbps: float) -> Optional[float]:
    """size_kb / bandwidth, with bandwidth given in GB/s (1 GB/s = 1e6 KB/s,
    decimal GB) so the result comes out in seconds."""
    if size_kb is None:
        return None
    return size_kb / (bandwidth_gbps * 1e6)


def analyze_experiment_efficiency(exp_dir: Path, timeout_s: float) -> Optional[Dict[str, Any]]:
    """One row of {total_iterations, total_time_s, bandwidth_latency_*,
    discard_ratio, dataset, opt, w_HW, seed, ...} for exp_dir, or None if its
    logs can't be loaded."""
    analyzer = ar.ResultsAnalyzer(exp_dir)
    if not analyzer.load_results():
        return None

    best_result = analyzer.get_best_result()
    best_iteration = best_result.iteration if best_result else None
    size_kb, size_source = best_model_size_kb(exp_dir, analyzer, best_iteration)

    total_time = analyzer.get_total_wall_time()
    if total_time is None:
        # No "TOTAL TIME" line: the run never finished on its own, i.e. it hit
        # the job's wall-clock limit instead of converging.
        total_time = timeout_s

    row = {
        "experiment": exp_dir.name,
        "best_nparams": best_result.nparams if best_result else None,
        "model_size_kb": size_kb,
        "model_size_source": size_source,
        "total_iterations": analyzer.get_total_iterations(),
        "total_time_s": total_time,
        "discard_ratio": analyzer.get_discard_ratio(),
    }
    for bw in _BANDWIDTHS_GBPS:
        tag = str(bw).replace(".", "_")
        row[f"bandwidth_latency_{tag}gbps_s"] = bandwidth_latency_seconds(size_kb, bw)

    for key, value in analyzer.config.items():
        if key not in row:
            row[key] = value
    return row


def collect_rows(exp_dirs: List[Path], timeout_s: float) -> List[Dict[str, Any]]:
    rows = []
    for exp_dir in exp_dirs:
        print(f"Analyzing: {exp_dir.name}")
        row = analyze_experiment_efficiency(exp_dir, timeout_s)
        if row is not None:
            rows.append(row)
    return rows


def build_dataframe(args) -> pd.DataFrame:
    exp_dirs = discover_experiment_dirs(args.parent_dir, args.experiment)
    if not exp_dirs:
        print("[ERROR] no data: pass --parent-dir and/or --experiment", file=sys.stderr)
        sys.exit(1)

    rows = collect_rows(exp_dirs, args.timeout_hours * 3600.0)
    if not rows:
        print("[ERROR] no experiment could be analyzed (missing algorithm_logs/acc_report.txt?)",
              file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    if "dataset" in df.columns:
        df["dataset"] = df["dataset"].str.lower()
    for col, _ in _METRICS + [("best_nparams", ""), ("model_size_kb", ""), ("w_HW", ""), ("seed", "")]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["module"] = df["mod_list"].apply(module_label) if "mod_list" in df.columns else "no-module"

    # config.yaml always defines w_HW (schema default), even when hardware_module
    # was never in mod_list -- in that case the value was never actually applied
    # during training, so blank it out rather than grouping on a meaningless default
    # (e.g. "no-module" and "flops-module" runs would otherwise look like they used
    # a real w_HW just because the field happens to carry its default).
    if "w_HW" in df.columns:
        hw_active = df["module"].isin(["hw-module", "flops+hw-module"])
        df.loc[~hw_active, "w_HW"] = float("nan")
    return df


def apply_filters(df: pd.DataFrame, args) -> pd.DataFrame:
    # dataset/opt/seed come from config.yaml (see analyze_experiment_efficiency) --
    # an experiment missing that file has none of them, so the column may not
    # exist in df at all. Warn and skip rather than KeyError in that case.
    if args.opt:
        if "opt" not in df.columns:
            print("[WARNING] --opt given but no experiment has an 'opt' (config.yaml missing?) -- ignoring filter")
        else:
            print(f"[DEBUG] args.opt: {args.opt}")
            opt = args.opt
            df = df[df["opt"].isin(opt)]
            print(f"[plan] {len(df)} experiment(s) after filtering by opt={opt}")
    if args.seed:
        if "seed" not in df.columns:
            print("[WARNING] --seed given but no experiment has a 'seed' (config.yaml missing?) -- ignoring filter")
        else:
            seed = args.seed
            df = df[df["seed"].isin([float(s) for s in seed])]
            print(f"[plan] {len(df)} experiment(s) after filtering by seed={seed}")
    if args.dataset:
        if "dataset" not in df.columns:
            print("[WARNING] --dataset given but no experiment has a 'dataset' (config.yaml missing?) -- ignoring filter")
        else:
            df = df[df["dataset"].str.lower().replace("-","").isin(args.dataset)]
            print(f"[plan] {len(df)} experiment(s) after filtering by dataset={args.dataset}")
    return df


def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std of total_time_s, both bandwidth latencies, and discard_ratio,
    grouped by (dataset, opt, module, w_HW) -- i.e. across whatever repeats
    (seeds) fall in each group. module (no-module/flops-module/hw-module/
    flops+hw-module) is included because w_HW alone can't tell a no-module run
    apart from a flops-module one (both blank it, see build_dataframe) -- so in
    practice you get one group per no-module, per flops-module, and one per
    distinct w_HW actually used by hardware_module (e.g. HW w=0.3 vs HW w=0.7).
    dataset/opt/w_HW come from config.yaml, so an experiment missing that file
    (see analyze_experiment_efficiency) has none of them -- those columns are
    only grouped on when at least one experiment actually has the value."""
    group_cols = [c for c in ("dataset", "opt", "module", "w_HW") if c in df.columns]
    records = []
    for keys, g in df.groupby(group_cols, dropna=False):
        rec = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        rec["n"] = len(g)
        for col, label in _METRICS:
            vals = g[col].dropna() if col in g.columns else pd.Series([], dtype=float)
            rec[f"{label}_n"] = int(vals.count())
            rec[f"{label}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            rec[f"{label}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
        records.append(rec)
    return pd.DataFrame(records).sort_values(group_cols).reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    """Factory for this script's argparse.ArgumentParser, kept separate from main()
    so tools (e.g. an interactive wizard) can introspect the available options
    without parsing sys.argv or running the analysis."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parent-dir", type=str, nargs="+", default=None,
                   help="Parent dir(s): every directory found anywhere underneath (at any depth) with an "
                        "algorithm_logs/ subfolder is analyzed as an experiment.")
    p.add_argument("--experiment", type=str, nargs="+", default=None,
                   help="Individual experiment dir(s) to analyze directly.")
    p.add_argument("--timeout-hours", type=float, default=_DEFAULT_TIMEOUT_HOURS,
                   help=f"Job wall-clock limit in hours (default: {_DEFAULT_TIMEOUT_HOURS:g}). Used as "
                        f"total_time_s for any experiment whose .out file has no 'TOTAL TIME' line, i.e. "
                        f"it never finished on its own and instead ran into this limit.")
    p.add_argument("--dataset", type=str, nargs="+", default=None, help="Restrict to these dataset(s). Default: all.")
    p.add_argument("--opt", type=str, nargs="+", default=None, help="Restrict to these opt(s). Default: all.")
    p.add_argument("--seed", type=str, nargs="+", default=None, help="Restrict to these seed(s). Default: all.")
    p.add_argument("--out-dir", type=str, default="experiment_efficiency_out")
    return p


def _unique_or_na(df: pd.DataFrame, col: str):
    """sorted(df[col].dropna().unique()), or 'n/a' if no experiment has that
    column at all (e.g. dataset/opt/w_HW when config.yaml is missing)."""
    return sorted(df[col].dropna().unique()) if col in df.columns else "n/a"


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.parent_dir and not args.experiment:
        parser.error("give --parent-dir and/or --experiment")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataframe(args)
    df = apply_filters(df, args)
    if df.empty:
        print("[ERROR] no rows left after filtering.", file=sys.stderr)
        sys.exit(1)

    print(f"\n[plan] {len(df)} experiment(s) after filtering -- "
          f"datasets={_unique_or_na(df, 'dataset')}, "
          f"opts={_unique_or_na(df, 'opt')}, "
          f"modules={_unique_or_na(df, 'module')}, "
          f"w_HW={_unique_or_na(df, 'w_HW')}")

    per_exp_path = out_dir / "experiment_efficiency.csv"
    df.to_csv(per_exp_path, index=False)
    print(f"\nPer-experiment CSV saved: {per_exp_path}")

    stats = compute_stats(df)
    stats_path = out_dir / "efficiency_stats_by_dataset_opt_wHW.csv"
    stats.to_csv(stats_path, index=False)
    print(f"Stats saved: {stats_path}\n")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
