"""Split results folders written by TWO roi_k_array runs (same fixed NAME, before
the date/job-id prefix): the first without modules, the second with
--mod_list flops_module, whose logs were appended to the first's.

For every folder under RESULTS_DIR it looks for the SLURM logs of the runs that
wrote it ($LOGS_DIR/roi_k_tuner_<job>_<task>.out, first line "task N: <NAME>"),
counts the iterations each one logged ("ACCURACY:" / "Chosen point:" lines)
and splits the files at that point into two new folders
    <date>_<jobid>_<name>        (date = .out modification time)
one per run. Split exactly (they are what analyze_results.py reads):
    acc_report.txt, score_report.txt, hyper-neural.txt, timing_report.csv,
    params_report.txt (no-module run only), flops_report.txt (flops run only),
    config.yaml (name updated; mod_list emptied for the first run), tuner.out link.
Model/, database/, symbolic/ and the other files were overwritten by the second
run: copied into its folder only. The symbolic logs (diagnosis/tuning/evidence)
are not split: they stay in the backup.
The original folder is MOVED (not deleted) to <RESULTS_DIR>_mixed_backup/.

Dry run by default; run it only after the second run has finished:
    python slurm/split_mixed_runs.py                 # shows what it would do
    python slurm/split_mixed_runs.py --apply
"""
import argparse
import csv
import os
import re
import shutil
import time
from pathlib import Path

import yaml

TASK_RE = re.compile(r"task \d+: (\S+)")
OUT_RE = re.compile(r"roi_k_tuner_(\d+)_(\d+)\.out$")
LINE_SPLIT = {"acc_report.txt": "ACCURACY:", "score_report.txt": "ACCURACY:",
              "hyper-neural.txt": "Chosen point:"}
NOT_SPLIT = {"diagnosis_symbolic_logs.txt", "tuning_symbolic_logs.txt", "evidence.txt"}


def runs_by_folder(logs_dir: Path):
    """{folder name: [(job id, .out path, n ACCURACY, n Chosen point), ...]} sorted by job id."""
    runs = {}
    for out in logs_dir.glob("roi_k_tuner_*.out"):
        m = OUT_RE.search(out.name)
        if not m:
            continue
        text = out.read_text(errors="replace")
        t = TASK_RE.search(text[:2000])
        if not t:
            continue
        n_acc = text.count("ACCURACY:")
        if n_acc == 0:          # crashed before the first iteration: wrote nothing
            continue
        runs.setdefault(Path(t.group(1)).name, []).append(
            (int(m.group(1)), out, n_acc, text.count("Chosen point:")))
    for v in runs.values():
        v.sort()
    return runs


def lines(p: Path):
    return p.read_text().splitlines(keepends=True) if p.exists() else []


def split_timing(src: Path, new_mods):
    """(header, rows of run 1, rows of run 2): run 2 starts where iteration restarts."""
    rows = list(csv.reader(src.open(newline="")))
    header, body = rows[0], rows[1:]
    cut = next((i for i in range(1, len(body)) if int(body[i][0]) <= int(body[i - 1][0])), len(body))
    new_header = header + [f"module_time_{m}" for m in new_mods if f"module_time_{m}" not in header]
    return header, body[:cut], new_header, body[cut:]


def write_csv(path: Path, header, rows):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def plan(exp: Path, runs):
    """Checks; returns (old, new) run tuples or raises ValueError with the reason."""
    if len(runs) != 2:
        raise ValueError(f"{len(runs)} SLURM runs with iterations found (need 2)")
    old, new = runs
    logs = exp / "algorithm_logs"
    n_acc = len(lines(logs / "acc_report.txt"))
    if n_acc != old[2] + new[2]:
        raise ValueError(f"acc_report has {n_acc} lines, .out files say {old[2]}+{new[2]} "
                         "(second run still going?)")
    n_par, n_flo = len(lines(logs / "params_report.txt")), len(lines(logs / "flops_report.txt"))
    if n_par != old[2] or n_flo != new[2]:
        raise ValueError(f"params_report {n_par} / flops_report {n_flo} lines, expected {old[2]} / {new[2]}: "
                         "first run is not the no-module one?")
    return old, new


def new_name(exp: Path, run):
    stamp = time.strftime("%y_%m_%d_%H", time.localtime(run[1].stat().st_mtime))
    return exp.parent / f"{stamp}_{run[0]}_{exp.name}"


def split(exp: Path, old, new, backup_dir: Path):
    logs = exp / "algorithm_logs"
    cfg = yaml.safe_load((exp / "config.yaml").read_text())
    new_mods = cfg.get("mod_list") or []
    dst = {"old": new_name(exp, old), "new": new_name(exp, new)}
    for d in dst.values():
        if d.exists():
            raise ValueError(f"{d} already exists")
        (d / "algorithm_logs").mkdir(parents=True)

    for fname, marker in LINE_SPLIT.items():
        ls = lines(logs / fname)
        if not ls:
            continue
        k = old[2] if marker == "ACCURACY:" else old[3]
        (dst["old"] / "algorithm_logs" / fname).write_text("".join(ls[:k]))
        (dst["new"] / "algorithm_logs" / fname).write_text("".join(ls[k:]))
    if (logs / "params_report.txt").exists():
        shutil.copy2(logs / "params_report.txt", dst["old"] / "algorithm_logs")
    for fname in ("flops_report.txt", "hardware_report.txt"):
        if (logs / fname).exists():
            shutil.copy2(logs / fname, dst["new"] / "algorithm_logs")
    if (logs / "timing_report.csv").exists():
        h_old, r_old, h_new, r_new = split_timing(logs / "timing_report.csv", new_mods)
        write_csv(dst["old"] / "algorithm_logs" / "timing_report.csv", h_old, r_old)
        write_csv(dst["new"] / "algorithm_logs" / "timing_report.csv", h_new, r_new)

    for key, run in (("old", old), ("new", new)):
        c = dict(cfg, name=str(dst[key]))
        if key == "old":
            c["mod_list"] = []
        (dst[key] / "config.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
        os.symlink(run[1], dst[key] / "tuner.out")

    # everything else (Model/, database/, symbolic/, ...) belongs to the last run
    for item in exp.iterdir():
        if item.name in ("algorithm_logs", "config.yaml") or item.suffix == ".out" \
                or item.name.endswith("_results.csv"):
            continue
        if item.is_dir():
            shutil.copytree(item, dst["new"] / item.name, symlinks=True)
        else:
            shutil.copy2(item, dst["new"] / item.name, follow_symlinks=False)

    backup_dir.mkdir(exist_ok=True)
    shutil.move(str(exp), str(backup_dir / exp.name))
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = Path(os.environ.get("WORK", ".")) / os.environ.get("USER", "") / "tuner"
    ap.add_argument("--results", type=Path, default=Path(os.environ.get("RESULTS_DIR", base / "results")))
    ap.add_argument("--logs", type=Path, default=Path(os.environ.get("LOGS_DIR", base / "logs")))
    ap.add_argument("--apply", action="store_true", help="really split (default: dry run)")
    args = ap.parse_args()

    backup_dir = args.results.parent / f"{args.results.name}_mixed_backup"
    runs = runs_by_folder(args.logs)
    print(f"results: {args.results}\nlogs:    {args.logs}\nbackup:  {backup_dir}\n")
    done = skipped = 0
    for exp in sorted(p for p in args.results.iterdir() if (p / "algorithm_logs").is_dir()):
        try:
            old, new = plan(exp, runs.get(exp.name, []))
        except ValueError as e:
            print(f"SKIP {exp.name}: {e}")
            skipped += 1
            continue
        print(f"{exp.name}\n  run 1: job {old[0]}, {old[2]} iterations -> {new_name(exp, old).name}"
              f"\n  run 2: job {new[0]}, {new[2]} iterations -> {new_name(exp, new).name}")
        if args.apply:
            split(exp, old, new, backup_dir)
        done += 1
    print(f"\n{done} folders {'split' if args.apply else 'to split (dry run, use --apply)'}, {skipped} skipped")


if __name__ == "__main__":
    main()
