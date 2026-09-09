"""
robustness_eval.py — operating-condition robustness of the two "final" models
(the ones handed to IHP for implementation): one ROI, one non-ROI.

The models were trained with a fixed number of frames, i.e. a *variable*
inter-frame time across recordings (short clip -> small dt, long clip -> large
dt). Here we measure how they behave at a *fixed* inter-frame time, swept over
1, 10, 50, 100, 200 ms, and draw "Accuracy vs inter-frame time" for both models
so their robustness can be compared.

Apple-to-apple across inter-frame times:
  * every test recording is cropped to its first WINDOW ms (default 1000 ms),
    so a long clip does not get more frames (and thus more votes) than a short
    one;
  * recordings shorter than WINDOW are DISCARDED (same sample set at every
    inter-frame time).

Per recording we keep:
  * the majority-vote prediction over the 1 s window  -> sequence accuracy;
  * the full per-frame prediction sequence            -> "stabilisation" frame,
    i.e. the index from which every later per-frame prediction is identical
    (no more oscillation). A model that stabilises at frame 2 rather than 4 is
    a better candidate for low-latency adaptive inference. We report both the
    stabilisation frame count and the stabilisation time (frame * dt).

Outputs (in --out-dir):
  robustness_accuracy_vs_interframe.png
  robustness_stabilisation.png
  robustness_summary.csv
  robustness_per_sample.csv

Usage:
  python robustness_eval.py \
      --plain-experiment 26_09_01_16_133015_gesture_fwdPass_64_2_300K \
      --roi-experiment   26_03_04_23_73623_roigesture_matrix_fwdPass_64_2_10000000000_flops_module \
      [--train] [--epochs N] [--model-name retrained-model.keras] \
      [--inter-frame-ms 1 10 50 100 200] [--window-ms 1000] [--out-dir robustness_out]
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

def _keep_indices_plain(dataset_path: str, window_us: int):
    """Indices (and labels) of test recordings with >= window_us of events."""
    import tonic

    raw = tonic.datasets.DVSGesture(save_to=dataset_path, train=False)
    keep, labels = [], []
    for i in range(len(raw)):
        try:
            events, y = raw[i]
        except Exception as exc:
            print(f"  [keep] skip {i}: {type(exc).__name__}: {exc}")
            continue
        if _duration_us(events) >= window_us:
            keep.append(i)
            labels.append(int(y))
    return keep, labels


def _plain_frames_at_dt(dataset_path, dt_us, window_us, keep_idx):
    """Yield [T, 64, 64, 2] float32 frame stacks for the kept test recordings."""
    import tonic
    import tonic.transforms as T

    tf = T.Compose([
        T.Denoise(filter_time=10000),
        T.Downsample(sensor_size=tonic.datasets.DVSGesture.sensor_size, target_size=(64, 64)),
        CropRelative(window_us),
        T.ToFrame(sensor_size=(64, 64, 2), time_window=int(dt_us)),
    ])
    ds = tonic.datasets.DVSGesture(save_to=dataset_path, train=False, transform=tf)
    for i in keep_idx:
        events, _ = ds[i]                                   # [T, 2, 64, 64]
        yield np.transpose(np.asarray(events), (0, 2, 3, 1)).astype(np.float32)


def _keep_indices_roi(dataset_path: str, frame_size: int, window_us: int):
    from components.gesture_dataset import DVSGestureROI

    raw = DVSGestureROI(dataset_path, output_size=(frame_size, frame_size, 2), train=False)
    keep, labels = [], []
    for i in range(len(raw)):
        try:
            x, y = raw[i]
        except Exception as exc:
            print(f"  [keep] skip {i}: {type(exc).__name__}: {exc}")
            continue
        if _duration_us(x["data"]) >= window_us:
            keep.append(i)
            labels.append(int(y))
    return keep, labels


def _roi_frames_at_dt(dataset_path, frame_size, dt_us, window_us, keep_idx):
    """Yield ([T, fs, fs, 2], [T, fs, fs, 1]) float32 stacks for kept recordings."""
    import tonic
    import tonic.transforms as T
    from components.gesture_dataset import DVSGestureROI

    ev_tf = T.Compose([
        T.Denoise(filter_time=10000),
        T.Downsample(sensor_size=(32, 32), target_size=(frame_size, frame_size)),
        CropRelative(window_us),
        T.ToFrame(sensor_size=(frame_size, frame_size, 2), time_window=int(dt_us)),
    ])
    pos_tf = T.Compose([
        T.Downsample(sensor_size=(128, 128, 1), target_size=(frame_size, frame_size)),
        CropRelative(window_us),
        T.ToFrame(sensor_size=(frame_size, frame_size, 1), time_window=int(dt_us)),
    ])
    ds = DVSGestureROI(
        dataset_path, output_size=(frame_size, frame_size, 2), train=False,
        transform=ev_tf, position_transform=pos_tf,
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


def _eval_at_dt(model, frame_iter, labels, is_roi: bool, dt_ms: float, n_classes: int = 11):
    """Run per-frame inference for every recording at this inter-frame time."""
    rows = []
    for k, (item, y_true) in enumerate(zip(frame_iter, labels)):
        if is_roi:
            data, pos = item
            logits = model([data, pos], training=False)
        else:
            data = item
            logits = model(data, training=False)
        logits = np.asarray(logits)                          # [T, C]
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

def _plot_accuracy(summ, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    models = list(dict.fromkeys(s["model"] for s in summ))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in models:
        rows = sorted((s for s in summ if s["model"] == m and s["n_samples"]),
                      key=lambda s: s["inter_frame_ms"])
        if not rows:
            continue
        x = [r["inter_frame_ms"] for r in rows]
        ax.plot(x, [r["vote_acc"] for r in rows], "-o", label=f"{m} — majority vote")
        ax.plot(x, [r["per_frame_acc"] for r in rows], "--x", alpha=0.6,
                label=f"{m} — per-frame")
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 50, 100, 200])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Inter-frame time (ms)")
    ax.set_ylabel("Accuracy (1 s window)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.set_title("Robustness to operating condition: accuracy vs inter-frame time")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _plot_stabilisation(summ, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    models = list(dict.fromkeys(s["model"] for s in summ))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for m in models:
        rows = sorted((s for s in summ if s["model"] == m and s["n_samples"]),
                      key=lambda s: s["inter_frame_ms"])
        if not rows:
            continue
        x = [r["inter_frame_ms"] for r in rows]
        for ax, key in ((axes[0], "stab_frame"), (axes[1], "stab_ms")):
            p50 = [r[f"{key}_p50"] for r in rows]
            p25 = [r[f"{key}_p25"] for r in rows]
            p75 = [r[f"{key}_p75"] for r in rows]
            line, = ax.plot(x, p50, "-o", label=m)
            ax.fill_between(x, p25, p75, alpha=0.15, color=line.get_color())
    for ax, ylab, ttl in ((axes[0], "Stabilisation frame (median, IQR)", "frames until stable"),
                          (axes[1], "Stabilisation time ms (median, IQR)", "time until stable")):
        ax.set_xscale("log")
        ax.set_xticks([1, 10, 50, 100, 200])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("Inter-frame time (ms)")
        ax.set_ylabel(ylab)
        ax.set_title(ttl)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Majority-vote stabilisation (earlier = better for low-latency adaptive inference)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _write_csvs(summ, per_sample, out_dir: Path):
    sfields = ["model", "inter_frame_ms", "n_samples", "vote_acc", "per_frame_acc",
               "mean_n_frames", "stab_frame_p25", "stab_frame_p50", "stab_frame_p75",
               "stab_ms_p25", "stab_ms_p50", "stab_ms_p75"]
    with open(out_dir / "robustness_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sfields)
        w.writeheader()
        for s in summ:
            w.writerow({k: s.get(k, "") for k in sfields})
    pfields = ["model", "inter_frame_ms", "sample", "true", "pred_vote", "correct",
               "n_frames", "per_frame_acc", "stab_frame", "stab_ms", "stable_label"]
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

def _run_experiment(experiment: Path, label: str, is_roi: bool, args):
    from exp_config import set_active_config, load_cfg

    config_path = experiment / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {experiment}")
    set_active_config(config_path)
    cfg = load_cfg(force=True)

    model = _resolve_model(experiment, args)

    window_us = int(args.window_ms * 1000)
    if is_roi:
        dataset_path = "rois_and_coordinates/datasets/"
        frame_size = _roi_frame_size(experiment)
        if not Path(dataset_path).exists():
            print(f"[{label}] ROI dataset '{dataset_path}' not found — skipping ROI model.")
            return [], []
        keep, labels = _keep_indices_roi(dataset_path, frame_size, window_us)
    else:
        dataset_path = "./data"
        keep, labels = _keep_indices_plain(dataset_path, window_us)

    print(f"[{label}] {len(keep)} test recordings with >= {args.window_ms} ms of events")
    if not keep:
        return [], []

    summ, per_sample = [], []
    for dt_ms in args.inter_frame_ms:
        dt_us = int(dt_ms * 1000)
        print(f"[{label}] inter-frame {dt_ms} ms (~{window_us // dt_us} frames/sample)...")
        if is_roi:
            it = _roi_frames_at_dt(dataset_path, frame_size, dt_us, window_us, keep)
        else:
            it = _plain_frames_at_dt(dataset_path, dt_us, window_us, keep)
        rows = _eval_at_dt(model, it, labels, is_roi, dt_ms, n_classes=cfg_n_classes(cfg))
        s = _summarise(rows, label, dt_ms)
        summ.append(s)
        for r in rows:
            per_sample.append({"model": label, "inter_frame_ms": dt_ms, **r})
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

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plain-experiment", type=str, default=None,
                   help="Experiment dir of the non-ROI model.")
    p.add_argument("--roi-experiment", type=str, default=None,
                   help="Experiment dir of the ROI model.")
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
    p.add_argument("--window-ms", type=float, default=1000.0,
                   help="Per-recording window kept for voting, in ms (default: 1000).")
    p.add_argument("--out-dir", type=str, default="robustness_out")
    args = p.parse_args()

    if not args.plain_experiment and not args.roi_experiment:
        p.error("give at least one of --plain-experiment / --roi-experiment")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summ, all_per_sample = [], []
    targets = []
    if args.plain_experiment:
        targets.append((Path(args.plain_experiment).resolve(), "non-ROI", False))
    if args.roi_experiment:
        targets.append((Path(args.roi_experiment).resolve(), "ROI", True))

    for experiment, label, is_roi in targets:
        print(f"\n{'='*70}\n{label}: {experiment}\n{'='*70}")
        summ, per_sample = _run_experiment(experiment, label, is_roi, args)
        all_summ.extend(summ)
        all_per_sample.extend(per_sample)

    if not all_summ:
        print("[ERROR] nothing evaluated.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*70}\nWriting outputs to {out_dir}/\n{'='*70}")
    _write_csvs(all_summ, all_per_sample, out_dir)
    _plot_accuracy(all_summ, out_dir / "robustness_accuracy_vs_interframe.png")
    _plot_stabilisation(all_summ, out_dir / "robustness_stabilisation.png")


if __name__ == "__main__":
    main()
