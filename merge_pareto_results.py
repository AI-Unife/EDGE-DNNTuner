"""Merge every per-experiment <name>_results.csv (written by analyze_results.py,
one row per tuner iteration) of the gesture / ROI runs into ONE csv, label the
six input configurations, flag the Pareto-optimal nets and plot the
accuracy-params and accuracy-FLOPs fronts.

Configurations (column `config`):
    gesture            dataset gesture              full frame, no ROI
    coords             roigesture_coords            ROI + centre of mass of the position map
    matrix in FC       roigesture_matrix, created_at <  2026-11-01   position matrix flattened into the FC
    matrix as channel  roigesture_matrix, created_at >= 2026-11-01   position matrix as an extra channel
    channels           roigesture_channels          ROI over delta_t/10 sub-windows as channels
    conv1d             roigesture_conv1d            ROI + Conv1D branch on the sub-ROI coordinates

Kept: fwdPass runs framed with delta_t on BOTH splits (train_framing and
test_framing == delta_t), every iteration with an accuracy (nets rejected by a
module have none) whose activation is relu or elu (ACTIVATIONS). All modules / epochs / thresholds are kept: filter on the
mod_list, epochs, nparams_th columns afterwards. The same experiment found in
two folders (copied twice) is kept once.

Pareto flags (computed within each config x delta_t; accuracy maximised,
params / FLOPs minimised):
    pareto_params  not dominated in (accuracy, nparams)
    pareto_flops   not dominated in (accuracy, flops)
    pareto_3d      not dominated in (accuracy, nparams, flops)

Usage (folders are searched recursively, e.g. the copies from Leonardo, the
unife cluster and the local disk):
    python merge_pareto_results.py DIR [DIR ...] [--out ../pareto_all_results.csv]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MATRIX_CUTOFF = pd.Timestamp("2026-11-01")
ACTIVATIONS = {"relu", "elu"}   # only nets with these activation functions are kept
CONFIGS = ["gesture", "coords", "matrix in FC", "matrix as channel", "channels", "conv1d"]
# dataviz reference categorical palette, fixed order (one slot per config) + a
# marker per config: with six series in a scatter colour alone is not enough.
COLORS = dict(zip(CONFIGS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]))
MARKERS = dict(zip(CONFIGS, ["o", "s", "^", "v", "D", "P"]))
FIRST_COLS = ["config", "delta_t_ms", "seed", "opt", "mod_list", "nparams_th", "epochs",
              "experiment", "source", "iteration", "accuracy", "nparams", "flops",
              "pareto_params", "pareto_flops", "pareto_3d"]


def config_label(dataset: str, created_at) -> str | None:
    if dataset == "gesture":
        return "gesture"
    if dataset == "roigesture_coords":
        return "coords"
    if dataset == "roigesture_matrix":
        return "matrix in FC" if created_at < MATRIX_CUTOFF else "matrix as channel"
    if dataset == "roigesture_channels":
        return "channels"
    if dataset == "roigesture_conv1d":
        return "conv1d"
    return None


# <yy_mm_dd_HH>_<jobid>_ prefix of the experiment folders (split halves included)
RUN_PREFIX = re.compile(r"^\d{2}_\d{2}_\d{2}_\d{2}_\d+_")


def load(roots):
    frames, skipped = [], {}
    seen = set()
    files = [f for root in roots for f in sorted(Path(root).rglob("*_results.csv"))
             if not f.name.startswith("summary_")]
    # Leonardo runs written twice into one folder (no prefix) were split by
    # slurm/split_mixed_runs.py into two prefixed halves: a CSV of the mixed
    # folder (analysed before the split) repeats their iterations -> dropped.
    prefixed = {RUN_PREFIX.sub("", f.name) for f in files if RUN_PREFIX.match(f.name)}

    def skip(reason, path):
        skipped.setdefault(reason, []).append(str(path))

    for f in files:
        # analyze_results.py writes <exp>_results.csv inside the experiment
        # folder, or flat in the output folder when one is given
        exp = f.name[: -len("_results.csv")]
        if not RUN_PREFIX.match(f.name) and f.name in prefixed:
            skip("mixed run already split into two halves", f)
            continue
        df = pd.read_csv(f, sep=";")
        if df.empty or "dataset" not in df:
            skip("empty or no config columns", f)
            continue
        first = df.iloc[0]
        created = pd.to_datetime(first.get("created_at"), errors="coerce")
        label = config_label(str(first["dataset"]), created)
        if label is None:
            skip(f"not a gesture dataset", f)
            continue
        if label.startswith("matrix") and pd.isna(created):
            skip("roigesture_matrix without created_at", f)
            continue
        if first.get("mode") != "fwdPass" or first.get("train_framing") != "delta_t" \
                or first.get("test_framing") != "delta_t":
            skip("not delta_t framing on both splits (or framing unknown)", f)
            continue
        if exp in seen:
            skip("duplicate experiment (kept the first copy)", f)
            continue
        seen.add(exp)
        if "activation" not in df:
            skip("no activation column", f)
            continue
        df = df[pd.to_numeric(df["accuracy"], errors="coerce").notna()
                & df["activation"].astype(str).str.lower().isin(ACTIVATIONS)].copy()
        if df.empty:
            skip(f"no evaluated net with activation in {sorted(ACTIVATIONS)}", f)
            continue
        df["config"], df["experiment"], df["source"] = label, exp, str(f.parent)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return data, skipped


def pareto_mask(acc: np.ndarray, costs: np.ndarray) -> np.ndarray:
    """True for points not dominated: no other point with acc >= and every cost <=,
    strictly better in at least one."""
    n = len(acc)
    keep = np.ones(n, bool)
    for i in range(n):
        ge = (acc >= acc[i]) & np.all(costs <= costs[i], axis=1)
        strict = (acc > acc[i]) | np.any(costs < costs[i], axis=1)
        if np.any(ge & strict):
            keep[i] = False
    return keep


def add_pareto(data: pd.DataFrame) -> pd.DataFrame:
    for c in ("accuracy", "nparams", "flops", "delta_t"):
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data["delta_t_ms"] = data["delta_t"] / 1000
    for col in ("pareto_params", "pareto_flops", "pareto_3d"):
        data[col] = False
    for _, g in data.groupby(["config", "delta_t_ms"]):
        acc = g["accuracy"].to_numpy()
        for col, costs in (("pareto_params", ["nparams"]), ("pareto_flops", ["flops"]),
                           ("pareto_3d", ["nparams", "flops"])):
            ok = g[costs].notna().all(axis=1).to_numpy()
            mask = np.zeros(len(g), bool)
            mask[ok] = pareto_mask(acc[ok], g[costs].to_numpy(float)[ok])
            data.loc[g.index, col] = mask
    return data


def plot(data: pd.DataFrame, out_png: Path):
    """One image per delta_t: <out>_dt<X>ms.png, accuracy vs params above accuracy vs FLOPs."""
    axes_spec = [("nparams", "pareto_params", "Parameters"), ("flops", "pareto_flops", "FLOPs")]
    for dt in sorted(data["delta_t_ms"].dropna().unique()):
        d = data[data["delta_t_ms"] == dt]
        fig, axes = plt.subplots(2, 1, figsize=(13, 13))
        for ax, (xcol, pcol, xlabel) in zip(axes, axes_spec):
            for cfg in CONFIGS:
                g = d[(d["config"] == cfg) & d[xcol].notna()]
                if g.empty:
                    continue
                ax.scatter(g[xcol], g["accuracy"], s=14, color=COLORS[cfg], alpha=0.45,
                           marker=MARKERS[cfg], linewidths=0)
                front = g[g[pcol]].sort_values(xcol)
                ax.step(front[xcol], front["accuracy"], where="post", color=COLORS[cfg], lw=2)
                ax.scatter(front[xcol], front["accuracy"], s=40, color=COLORS[cfg],
                           marker=MARKERS[cfg], edgecolors="white", linewidths=1.5, zorder=3,
                           label=cfg)
            if xcol == "nparams":
                ax.axvline(300_000, color="#e34948", lw=1.5, ls="--", zorder=4)
                ax.annotate("300K", xy=(300_000, 1), xycoords=("data", "axes fraction"),
                            xytext=(4, -4), textcoords="offset points", va="top",
                            color="#e34948", fontsize=11)
            ax.set_xscale("log")
            ax.grid(True, which="major", color="#e4e3dd", lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.set_ylim(0, 1)
            ax.set_yticks(np.arange(0, 1.01, 0.1))
            ax.tick_params(labelsize=11)
        handles = {}
        for ax in axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(l, h)
        order = [c for c in CONFIGS if c in handles]
        fig.suptitle(f"Pareto fronts, \u0394t = {dt:g} ms "
                     "(points: every evaluated net; lines: front per configuration)",
                     y=0.995, fontsize=13)
        fig.legend([handles[c] for c in order], order, loc="upper center", ncol=len(order),
                   frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.975))
        fig.tight_layout(rect=(0, 0, 1, 0.945), h_pad=2.5)
        png = out_png.with_name(f"{out_png.stem}_dt{dt:g}ms.png")
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"plot  -> {png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=Path, help="folders searched recursively for *_results.csv")
    ap.add_argument("--out", type=Path, default=Path("../pareto_all_results.csv"))
    args = ap.parse_args()

    data, skipped = load(args.roots)
    for reason, files in skipped.items():
        print(f"skipped {len(files):3d}: {reason}")
        for f in files[:5]:
            print(f"           {f}")
        if len(files) > 5:
            print(f"           ... (+{len(files) - 5})")
    if data.empty:
        raise SystemExit("no gesture/ROI delta_t results found")

    data = add_pareto(data)
    cols = [c for c in FIRST_COLS if c in data] + [c for c in data if c not in FIRST_COLS]
    data = data[cols].sort_values(["config", "delta_t_ms", "experiment", "iteration"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out, sep=";", index=False)
    print(f"\ncsv   -> {args.out}  ({len(data)} iterations, {data['experiment'].nunique()} runs)")
    print(data.groupby(["config", "delta_t_ms"])["experiment"].nunique().unstack(fill_value=0)
          .to_string())
    plot(data, args.out.with_suffix(".png"))


if __name__ == "__main__":
    main()
