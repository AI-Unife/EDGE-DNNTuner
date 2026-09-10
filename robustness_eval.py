"""
robustness_eval.py — operating-condition robustness of the "final" models
(the ones handed to IHP for implementation): every ROI experiment in a parent
folder, compared against a single non-ROI (gesture) experiment.

The models were trained with a fixed number of frames, i.e. a *variable*
inter-frame time across recordings (short clip -> small dt, long clip -> large
dt). Here we measure how they behave at a *fixed* inter-frame time, swept over
1, 10, 50, 100, 200 ms, and draw "Accuracy vs inter-frame time" so ROI vs
non-ROI robustness can be compared.

Two window regimes, run back-to-back (``--window-ms 1000 0``):
  * 1 s window — every test recording is cropped to its first 1000 ms and
    recordings shorter than that are DISCARDED. This is the apple-to-apple
    comparison across inter-frame times (a long clip cannot get more votes than
    a short one).
  * no limit (``0``) — no crop, no discard: the full recording is used.

Per recording we keep:
  * the majority-vote prediction over the window  -> sequence accuracy;
  * the full per-frame prediction sequence        -> "stabilisation" frame,
    i.e. the index from which every later per-frame prediction is identical
    (no more oscillation). A model that stabilises at frame 2 rather than 4 is
    a better candidate for low-latency adaptive inference. We report both the
    stabilisation frame count and the stabilisation time (frame * dt).

Outputs (in --out-dir), one plot pair per window regime:
  robustness_accuracy_<window>.png
  robustness_stabilisation_<window>.png
  robustness_summary.csv        (kind, model, window, inter_frame_ms, ...)
  robustness_per_sample.csv

Usage:
  python robustness_eval.py \
      --plain-experiment 26_09_01_16_133015_gesture_fwdPass_64_2_300K \
      --roi-parent       results_gesture/roi_matrix \
      [--roi-experiment ONE_MORE_DIR] \
      [--train] [--epochs N] [--model-name retrained-model.keras] \
      [--inter-frame-ms 1 10 50 100 200] [--window-ms 1000 0] [--out-dir robustness_out]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Event transforms
# ---------------------------------------------------------------------------

class CropRelative:
    """Keep only the events in the first ``window_us`` microseconds of a recording
    (relative to its own first timestamp)."""

    def __init__(self, window_us: int):
        self.window_us = int(window_us)

    def __call__(self, events):
        t = events["t"]
        if len(t) == 0:
            return events
        return events[t - t[0] <= self.window_us]


def _duration_us(events) -> float:
    t = events["t"]
    return float(t[-1] - t[0]) if len(t) else 0.0


# ---------------------------------------------------------------------------
# Model resolution / (re)training
# ---------------------------------------------------------------------------

def _resolve_model(experiment: Path, args):
    """Return a loaded Keras model for ``experiment``; retrain from the logs when
    the checkpoint is missing and --train was given."""
    from test_utils import load_keras_model

    model_path = experiment / "Model" / args.model_name
    if not model_path.exists() and args.train:
        print(f"[{experiment.name}] {args.model_name} missing -> retraining from logs...")
        from test_experiment import run_single_experiment
        ns = argparse.Namespace(
            experiment=str(experiment),
            epochs=args.epochs,
            backend=None,
            delta_t=args.train_delta_t,
            selection_mode="auto_activation_retrain",
            force_winner_activation=False,
        )
        run_single_experiment(experiment, ns, "auto_activation_retrain", "relu")

    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found. Run with --train to rebuild+retrain it, "
            f"or point --model-name at an existing checkpoint."
        )
    print(f"[{experiment.name}] loading {model_path}")
    return load_keras_model(str(model_path))


# ---------------------------------------------------------------------------
# Test-set builders (ragged: one [T_i, ...] stack per recording)
# ---------------------------------------------------------------------------

def _keep_indices_plain(dataset_path: str, window_us):
    """Indices (and labels) of usable test recordings. When ``window_us`` is not
    None, drop recordings shorter than that (so the sample set is identical at
    every inter-frame time); when None, keep every readable recording."""
    import tonic

    raw = tonic.datasets.DVSGesture(save_to=dataset_path, train=False)
    keep, labels = [], []
    for i in range(len(raw)):
        try:
            events, y = raw[i]
        except Exception as exc:
            print(f"  [keep] skip {i}: {type(exc).__name__}: {exc}")
            continue
        if window_us is None or _duration_us(events) >= window_us:
            keep.append(i)
            labels.append(int(y))
    return keep, labels


def _plain_frames_at_dt(dataset_path, dt_us, window_us, keep_idx):
    """Yield [T, 64, 64, 2] float32 frame stacks for the kept test recordings."""
    import tonic
    import tonic.transforms as T

    steps = [
        T.Denoise(filter_time=10000),
        T.Downsample(sensor_size=tonic.datasets.DVSGesture.sensor_size, target_size=(64, 64)),
    ]
    if window_us is not None:
        steps.append(CropRelative(window_us))
    steps.append(T.ToFrame(sensor_size=(64, 64, 2), time_window=int(dt_us)))
    ds = tonic.datasets.DVSGesture(save_to=dataset_path, train=False, transform=T.Compose(steps))
    for i in keep_idx:
        events, _ = ds[i]                                   # [T, 2, 64, 64]
        yield np.transpose(np.asarray(events), (0, 2, 3, 1)).astype(np.float32)


def _keep_indices_roi(dataset_path: str, frame_size: int, window_us):
    from components.gesture_dataset import DVSGestureROI

    raw = DVSGestureROI(dataset_path, output_size=(frame_size, frame_size, 2), train=False)
    keep, labels = [], []
    for i in range(len(raw)):
        try:
            x, y = raw[i]
        except Exception as exc:
            print(f"  [keep] skip {i}: {type(exc).__name__}: {exc}")
            continue
        if window_us is None or _duration_us(x["data"]) >= window_us:
            keep.append(i)
            labels.append(int(y))
    return keep, labels


def _roi_frames_at_dt(dataset_path, frame_size, dt_us, window_us, keep_idx):
    """Yield ([T, fs, fs, 2], [T, fs, fs, 1]) float32 stacks for kept recordings."""
    import tonic
    import tonic.transforms as T
    from components.gesture_dataset import DVSGestureROI

    ev_steps = [
        T.Denoise(filter_time=10000),
        T.Downsample(sensor_size=(32, 32), target_size=(frame_size, frame_size)),
    ]
    pos_steps = [T.Downsample(sensor_size=(128, 128, 1), target_size=(frame_size, frame_size))]
    if window_us is not None:
        ev_steps.append(CropRelative(window_us))
        pos_steps.append(CropRelative(window_us))
    ev_steps.append(T.ToFrame(sensor_size=(frame_size, frame_size, 2), time_window=int(dt_us)))
    pos_steps.append(T.ToFrame(sensor_size=(frame_size, frame_size, 1), time_window=int(dt_us)))

    ds = DVSGestureROI(
        dataset_path, output_size=(frame_size, frame_size, 2), train=False,
        transform=T.Compose(ev_steps), position_transform=T.Compose(pos_steps),
    )
    for i in keep_idx:
        x, _ = ds[i]
        data = np.transpose(np.asarray(x["data"]), (0, 2, 3, 1)).astype(np.float32)  # [T, fs, fs, 2]
        pos = np.transpose(np.asarray(x["pos"]), (0, 2, 3, 1)).astype(np.float32)    # [T, fs, fs, 1]
        n = min(len(data), len(pos))
        yield data[:n], pos[:n]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _stabilisation_frame(seq: np.ndarray) -> int:
    """Index from which every later per-frame prediction is identical.
    0 = stable from the first frame; len-1 = only the last frame is 'stable'."""
    if len(seq) <= 1:
        return 0
    changes = np.nonzero(seq[1:] != seq[:-1])[0]
    return int(changes[-1] + 1) if len(changes) else 0


def _infer_frames(model, data, pos, is_roi, chunk: int = 512):
    """Per-frame logits [T, C], sub-batched so a long stack does not blow memory."""
    out = []
    for s in range(0, len(data), chunk):
        d = data[s:s + chunk]
        if is_roi:
            out.append(np.asarray(model([d, pos[s:s + chunk]], training=False)))
        else:
            out.append(np.asarray(model(d, training=False)))
    return np.concatenate(out, axis=0) if out else np.zeros((0, 1), np.float32)


def _eval_at_dt(model, frame_iter, labels, is_roi: bool, dt_ms: float, n_classes: int = 11):
    """Run per-frame inference for every recording at this inter-frame time."""
    rows = []
    for k, (item, y_true) in enumerate(zip(frame_iter, labels)):
        if is_roi:
            data, pos = item
        else:
            data, pos = item, None
        logits = _infer_frames(model, data, pos, is_roi)     # [T, C]
        seq = logits.argmax(axis=1).astype(np.int64)         # [T]
        if len(seq) == 0:
            continue
        vote = int(np.bincount(seq, minlength=n_classes).argmax())
        k_stab = _stabilisation_frame(seq)
        rows.append({
            "sample": k,
            "true": int(y_true),
            "pred_vote": vote,
            "correct": int(vote == y_true),
            "n_frames": int(len(seq)),
            "per_frame_acc": float(np.mean(seq == y_true)),
            "stab_frame": k_stab,
            "stab_ms": float(k_stab * dt_ms),
            "stable_label": int(seq[-1]),
        })
    return rows


def _summarise(rows, model_name, dt_ms):
    if not rows:
        return {"model": model_name, "inter_frame_ms": dt_ms, "n_samples": 0}
    sf = np.array([r["stab_frame"] for r in rows], float)
    sm = np.array([r["stab_ms"] for r in rows], float)
    return {
        "model": model_name,
        "inter_frame_ms": dt_ms,
        "n_samples": len(rows),
        "vote_acc": float(np.mean([r["correct"] for r in rows])),
        "per_frame_acc": float(np.mean([r["per_frame_acc"] for r in rows])),
        "mean_n_frames": float(np.mean([r["n_frames"] for r in rows])),
        "stab_frame_p25": float(np.percentile(sf, 25)),
        "stab_frame_p50": float(np.percentile(sf, 50)),
        "stab_frame_p75": float(np.percentile(sf, 75)),
        "stab_ms_p25": float(np.percentile(sm, 25)),
        "stab_ms_p50": float(np.percentile(sm, 50)),
        "stab_ms_p75": float(np.percentile(sm, 75)),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _short(name: str, n: int = 28) -> str:
    return name if len(name) <= n else "…" + name[-(n - 1):]


def _plot_accuracy(summ, out_path: Path, wtag: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    models = list(dict.fromkeys(s["model"] for s in summ))
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in models:
        rows = sorted((s for s in summ if s["model"] == m and s["n_samples"]),
                      key=lambda s: s["inter_frame_ms"])
        if not rows:
            continue
        kind = rows[0].get("kind", "")
        x = [r["inter_frame_ms"] for r in rows]
        line, = ax.plot(x, [r["vote_acc"] for r in rows], "-o",
                        label=f"{kind} · {_short(m)} — vote")
        ax.plot(x, [r["per_frame_acc"] for r in rows], "--x", alpha=0.5,
                color=line.get_color(), label=f"{kind} · {_short(m)} — per-frame")
    ax.set_xscale("log")
    ax.set_xticks(sorted({r["inter_frame_ms"] for r in summ}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Inter-frame time (ms)")
    ax.set_ylabel(f"Accuracy ({wtag} window)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Robustness to operating condition — accuracy vs inter-frame time ({wtag})")
    ax.legend(fontsize=7, ncol=1, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _plot_stabilisation(summ, out_path: Path, wtag: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    models = list(dict.fromkeys(s["model"] for s in summ))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for m in models:
        rows = sorted((s for s in summ if s["model"] == m and s["n_samples"]),
                      key=lambda s: s["inter_frame_ms"])
        if not rows:
            continue
        kind = rows[0].get("kind", "")
        x = [r["inter_frame_ms"] for r in rows]
        for ax, key in ((axes[0], "stab_frame"), (axes[1], "stab_ms")):
            p50 = [r[f"{key}_p50"] for r in rows]
            p25 = [r[f"{key}_p25"] for r in rows]
            p75 = [r[f"{key}_p75"] for r in rows]
            line, = ax.plot(x, p50, "-o", label=f"{kind} · {_short(m)}")
            ax.fill_between(x, p25, p75, alpha=0.15, color=line.get_color())
    ticks = sorted({r["inter_frame_ms"] for r in summ})
    for ax, ylab, ttl in ((axes[0], "Stabilisation frame (median, IQR)", "frames until stable"),
                          (axes[1], "Stabilisation time ms (median, IQR)", "time until stable")):
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("Inter-frame time (ms)")
        ax.set_ylabel(ylab)
        ax.set_title(ttl)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"Majority-vote stabilisation ({wtag}) — earlier = better for low-latency adaptive inference")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _write_csvs(summ, per_sample, out_dir: Path):
    sfields = ["kind", "model", "window", "inter_frame_ms", "n_samples", "vote_acc",
               "per_frame_acc", "mean_n_frames", "stab_frame_p25", "stab_frame_p50",
               "stab_frame_p75", "stab_ms_p25", "stab_ms_p50", "stab_ms_p75"]
    with open(out_dir / "robustness_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sfields)
        w.writeheader()
        for s in summ:
            w.writerow({k: s.get(k, "") for k in sfields})
    pfields = ["kind", "model", "window", "inter_frame_ms", "sample", "true", "pred_vote",
               "correct", "n_frames", "per_frame_acc", "stab_frame", "stab_ms", "stable_label"]
    with open(out_dir / "robustness_per_sample.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pfields)
        w.writeheader()
        for r in per_sample:
            w.writerow({k: r.get(k, "") for k in pfields})
    print(f"  wrote {out_dir/'robustness_summary.csv'}")
    print(f"  wrote {out_dir/'robustness_per_sample.csv'}")


# ---------------------------------------------------------------------------
# Per-experiment sweep
# ---------------------------------------------------------------------------

def _run_experiment(experiment: Path, kind: str, is_roi: bool, args, windows):
    """Sweep one experiment over every window and every inter-frame time."""
    from exp_config import set_active_config, load_cfg

    config_path = experiment / "config.yaml"
    if not config_path.exists():
        print(f"[{experiment.name}] no config.yaml — skipping")
        return [], []
    set_active_config(config_path)
    cfg = load_cfg(force=True)

    model = _resolve_model(experiment, args)
    label = experiment.name
    n_classes = cfg_n_classes(cfg)

    if is_roi:
        dataset_path = "rois_and_coordinates/datasets/"
        if not Path(dataset_path).exists():
            print(f"[{label}] ROI dataset '{dataset_path}' not found — skipping.")
            return [], []
        frame_size = _roi_frame_size(experiment)
    else:
        dataset_path, frame_size = "./data", None

    summ, per_sample = [], []
    for window_us in windows:
        wtag = "full" if window_us is None else f"{int(round(window_us / 1000))}ms"
        if is_roi:
            keep, labels = _keep_indices_roi(dataset_path, frame_size, window_us)
        else:
            keep, labels = _keep_indices_plain(dataset_path, window_us)
        print(f"[{label}] window={wtag}: {len(keep)} usable test recordings")
        if not keep:
            continue

        for dt_ms in args.inter_frame_ms:
            dt_us = int(dt_ms * 1000)
            approx = "variable" if window_us is None else f"~{window_us // dt_us}"
            print(f"[{label}] window={wtag}  inter-frame {dt_ms} ms ({approx} frames/sample)...")
            if is_roi:
                it = _roi_frames_at_dt(dataset_path, frame_size, dt_us, window_us, keep)
            else:
                it = _plain_frames_at_dt(dataset_path, dt_us, window_us, keep)
            rows = _eval_at_dt(model, it, labels, is_roi, dt_ms, n_classes=n_classes)
            s = _summarise(rows, label, dt_ms)
            s["window"] = wtag
            s["kind"] = kind
            summ.append(s)
            for r in rows:
                per_sample.append({"kind": kind, "model": label, "window": wtag,
                                   "inter_frame_ms": dt_ms, **r})
            if s.get("n_samples"):
                print(f"    vote_acc={s['vote_acc']:.4f}  per_frame_acc={s['per_frame_acc']:.4f}  "
                      f"stab_frame(med)={s['stab_frame_p50']:.1f}  stab_ms(med)={s['stab_ms_p50']:.1f}")
    return summ, per_sample


def cfg_n_classes(cfg) -> int:
    return int(getattr(cfg, "n_classes", 11) or 11)


def _roi_frame_size(experiment: Path) -> int:
    import re
    pat = re.compile(r"input_shape=\((\d+)")
    for out_file in sorted(experiment.glob("*.out")):
        m = pat.search(out_file.read_text(errors="replace"))
        if m:
            return int(m.group(1))
    return 32


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _roi_experiment_dirs(args):
    dirs = []
    if args.roi_experiment:
        dirs.append(Path(args.roi_experiment).resolve())
    if args.roi_parent:
        parent = Path(args.roi_parent).resolve()
        if not parent.is_dir():
            print(f"[ERROR] --roi-parent not a directory: {parent}", file=sys.stderr)
            sys.exit(1)
        for d in sorted(parent.iterdir()):
            if d.is_dir() and (d / "config.yaml").exists() and list(d.glob("*.out")):
                dirs.append(d.resolve())
    # de-dup, keep order
    seen, out = set(), []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plain-experiment", type=str, default=None,
                   help="Experiment dir of the single non-ROI (gesture) model.")
    p.add_argument("--roi-experiment", type=str, default=None,
                   help="A single ROI experiment dir.")
    p.add_argument("--roi-parent", type=str, default=None,
                   help="Parent dir: every subfolder containing config.yaml + a *.out file "
                        "is treated as an ROI experiment and evaluated.")
    p.add_argument("--model-name", type=str, default="retrained-model.keras",
                   help="Checkpoint file under <experiment>/Model/ (default: retrained-model.keras).")
    p.add_argument("--train", action="store_true",
                   help="If the checkpoint is missing, rebuild the winning architecture "
                        "from the logs and retrain it (auto_activation_retrain).")
    p.add_argument("--epochs", type=int, default=None, help="Override epochs when --train retrains.")
    p.add_argument("--train-delta-t", type=int, default=None, dest="train_delta_t",
                   help="delta_t (us) written to config before a --train retrain.")
    p.add_argument("--inter-frame-ms", type=float, nargs="+", default=[1, 10, 50, 100, 200],
                   help="Inter-frame times to sweep, in ms (default: 1 10 50 100 200).")
    p.add_argument("--window-ms", type=float, nargs="+", default=[1000, 0],
                   help="Windows to evaluate, in ms. 0 = no limit (full recording, no sample "
                        "discard). Default: 1000 0 (both the 1 s window and the unlimited case).")
    p.add_argument("--out-dir", type=str, default="robustness_out")
    args = p.parse_args()

    roi_dirs = _roi_experiment_dirs(args)
    if not args.plain_experiment and not roi_dirs:
        p.error("give --plain-experiment and/or --roi-experiment / --roi-parent")

    windows = [None if w == 0 else int(round(w * 1000)) for w in args.window_ms]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = []
    if args.plain_experiment:
        targets.append((Path(args.plain_experiment).resolve(), "non-ROI", False))
    for d in roi_dirs:
        targets.append((d, "ROI", True))

    print(f"[plan] {len(targets)} experiment(s), windows={['full' if w is None else f'{w//1000}ms' for w in windows]}, "
          f"inter-frame ms={args.inter_frame_ms}")

    all_summ, all_per_sample = [], []
    for experiment, kind, is_roi in targets:
        print(f"\n{'='*70}\n{kind}: {experiment}\n{'='*70}")
        try:
            summ, per_sample = _run_experiment(experiment, kind, is_roi, args, windows)
        except Exception as exc:
            print(f"[ERROR] {experiment}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        all_summ.extend(summ)
        all_per_sample.extend(per_sample)

    if not all_summ:
        print("[ERROR] nothing evaluated.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*70}\nWriting outputs to {out_dir}/\n{'='*70}")
    _write_csvs(all_summ, all_per_sample, out_dir)
    for wtag in dict.fromkeys(s["window"] for s in all_summ):
        sub = [s for s in all_summ if s["window"] == wtag]
        _plot_accuracy(sub, out_dir / f"robustness_accuracy_{wtag}.png", wtag)
        _plot_stabilisation(sub, out_dir / f"robustness_stabilisation_{wtag}.png", wtag)


if __name__ == "__main__":
    main()
