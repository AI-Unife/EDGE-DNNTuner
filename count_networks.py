"""
count_networks.py — count how many networks the tuner tested, per dataset.

Walks a folder (recursively) looking for SLURM/tuner ".out" logs. In each log:
  - every "START TRAINING ITERATION <n>" line is one network proposed by the
    optimizer (controller.training());
  - a proposal followed by "Constraint Violated" (FLOPs / params / latency over
    threshold) is discarded without training;
  - every other proposal is a network that was actually trained and tested.

The dataset (and any other --group-by field) is read from the config.yaml of the
experiment, i.e. the closest config.yaml found in the .out's folder or above it.

Usage:
    python count_networks.py <folder>
    python count_networks.py <folder> --group-by dataset mode delta_t
    python count_networks.py <folder> --per-experiment --csv networks.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

START_RE = re.compile(r"START TRAINING ITERATION\s+\d+")
VIOLATION = "Constraint Violated"


def find_config(out_file: Path, root: Path) -> Path | None:
    """Closest config.yaml in the .out's folder or any parent, up to root."""
    for d in [out_file.parent, *out_file.parent.parents]:
        cfg = d / "config.yaml"
        if cfg.is_file():
            return cfg
        if d == root:
            break
    return None


def read_config(cfg_path: Path | None) -> dict:
    if cfg_path is None:
        return {}
    try:
        import yaml
        return yaml.safe_load(cfg_path.read_text(errors="replace")) or {}
    except Exception:
        # Fallback: plain "key: value" lines
        out = {}
        for line in cfg_path.read_text(errors="replace").splitlines():
            m = re.match(r"^(\w+):\s*(.*)$", line)
            if m:
                out[m.group(1)] = m.group(2).strip()
        return out


def count_out(out_file: Path) -> tuple[int, int]:
    """(proposed, discarded) networks in one .out log."""
    proposed = discarded = 0
    violated_current = False
    with open(out_file, errors="replace") as f:
        for line in f:
            if START_RE.search(line):
                proposed += 1
                violated_current = False
            elif VIOLATION in line and proposed and not violated_current:
                # FLOPs and latency violations can both be printed for the same network
                discarded += 1
                violated_current = True
    return proposed, discarded


def main() -> None:
    p = argparse.ArgumentParser(description="Count networks tested by the tuner, per dataset.")
    p.add_argument("folder", type=str, help="Root folder to scan recursively for .out files")
    p.add_argument("--group-by", nargs="+", default=["dataset"],
                   help="config.yaml fields to group by (default: dataset), e.g. dataset mode delta_t")
    p.add_argument("--per-experiment", action="store_true",
                   help="Also print one line per .out file")
    p.add_argument("--csv", type=str, default=None,
                   help="Optional path of a CSV with one row per .out file")
    args = p.parse_args()

    root = Path(args.folder).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"Not a folder: {root}")

    out_files = sorted(root.rglob("*.out"))
    if not out_files:
        sys.exit(f"No .out files under {root}")

    totals = defaultdict(lambda: {"experiments": set(), "out_files": 0,
                                  "proposed": 0, "trained": 0, "discarded": 0})
    rows = []
    for out_file in out_files:
        proposed, discarded = count_out(out_file)
        cfg_path = find_config(out_file, root)
        cfg = read_config(cfg_path)
        key = tuple(str(cfg.get(k, "?")) for k in args.group_by)
        exp_dir = cfg_path.parent if cfg_path else out_file.parent

        t = totals[key]
        t["experiments"].add(exp_dir)
        t["out_files"] += 1
        t["proposed"] += proposed
        t["discarded"] += discarded
        t["trained"] += proposed - discarded
        rows.append({
            **{k: v for k, v in zip(args.group_by, key)},
            "out_file": str(out_file.relative_to(root)),
            "proposed": proposed, "trained": proposed - discarded, "discarded": discarded,
            "has_config": cfg_path is not None,
        })

    if args.per_experiment:
        print("\nPer .out file:")
        for r in rows:
            label = " / ".join(r[k] for k in args.group_by)
            print(f"  {label:<30} trained={r['trained']:>5}  discarded={r['discarded']:>5}  {r['out_file']}")

    header = " / ".join(args.group_by)
    width = max(len(header), *(len(" / ".join(k)) for k in totals))
    print(f"\n{header:<{width}}  {'exps':>5}  {'.out':>5}  {'trained':>8}  {'discarded':>9}  {'proposed':>8}")
    print("-" * (width + 46))
    grand = {"experiments": 0, "out_files": 0, "proposed": 0, "trained": 0, "discarded": 0}
    for key in sorted(totals):
        t = totals[key]
        n_exp = len(t["experiments"])
        print(f"{' / '.join(key):<{width}}  {n_exp:>5}  {t['out_files']:>5}  "
              f"{t['trained']:>8}  {t['discarded']:>9}  {t['proposed']:>8}")
        grand["experiments"] += n_exp
        for k in ("out_files", "proposed", "trained", "discarded"):
            grand[k] += t[k]
    print("-" * (width + 46))
    print(f"{'TOTAL':<{width}}  {grand['experiments']:>5}  {grand['out_files']:>5}  "
          f"{grand['trained']:>8}  {grand['discarded']:>9}  {grand['proposed']:>8}")

    no_cfg = sum(not r["has_config"] for r in rows)
    if no_cfg:
        print(f"\n[note] {no_cfg} .out file(s) without a config.yaml above them are grouped under '?'.")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
