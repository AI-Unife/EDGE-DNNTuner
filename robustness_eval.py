"""
robustness_eval.py — operating-condition robustness of the "final" models
(the ones handed to IHP for implementation): one or more ROI experiments
(a single --roi-experiment, an entire --roi-parent folder, or both — either
roigesture_matrix or roigesture_coords, auto-detected per experiment from its
config.yaml and labelled "ROI-matrix" / "ROI-coords" accordingly), compared
against a single non-ROI (gesture) --plain-experiment.

Re-running the script only computes what is missing: any (kind, model) whose
full requested window/inter-frame matrix is already in
<out-dir>/robustness_summary.csv is reused as-is instead of recomputed — most
usefully the non-ROI model, which almost never changes between runs while new
ROI experiments get added to the parent folder.

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

Optional extra analysis (--time-profile-dt-ms X): FIX the inter-frame time to
X ms and instead ask "how does accuracy evolve as more time/frames are
observed?" — a running majority vote after 1, 2, 3, ... frames, one curve per
model. The underlying computation ALWAYS uses each recording UNCROPPED and
only up to its own actual duration (no population filtering, no discard by
length) — --window-ms never changes which recordings or which frames are
used here. It is computed in three ways from that same inference pass:

  * drop-out (robustness_time_profile.csv): a recording simply stops
    contributing once its own frames run out, so later checkpoints average
    over fewer, longer-lasting recordings — n_samples says how many at each
    point. The honest, non-extrapolated view; its tail gets noisy as
    n_samples shrinks. Companion of the stabilisation metric above:
    stabilisation asks *when* the vote stops oscillating, this asks
    *whether* it is correct at each point in time.
  * carry-forward (robustness_time_profile_carryforward.csv): same x-axis,
    but a finished recording keeps contributing its LAST vote to every later
    checkpoint instead of dropping out, so n_samples stays constant (= every
    recording) throughout — comparable point-to-point, at the cost of
    "freezing" recordings that have no more real evidence to give. Its very
    last checkpoint (elapsed = the longest recording's own duration) is
    numerically identical to the 'robustness' analysis's full-window
    vote_acc for this dt, since by then every recording is contributing its
    own true full-length vote.

--window-ms (same list as above) additionally TRUNCATES the output of these
two elapsed-time curves: each finite value keeps only elapsed_ms <= window,
tagged with that window instead of "full" — an exact prefix of the "full"
curve (same population, same values), not a different, cropped/filtered
computation. Both write one PNG per requested window:
robustness_accuracy_over_time_dt<X>ms_<window>.png and
..._carryforward.png. All of it is cached independently of the main sweep,
per (kind, model, window, dt).

Alongside it, the SAME inference pass also produces a %-of-own-duration
version (--time-profile-pct-steps, default 100 steps), UNAFFECTED by
--window-ms: accuracy at 1%, 2%, ..., 100% of EACH recording's own length,
instead of at an absolute elapsed time. Every recording contributes at every
percentage checkpoint (mapped to its own nearest frame), so n_samples stays
constant across the curve, and the 100% point is exactly "each recording's
own full-length majority vote, averaged over all of them" — the same
quantity as the carry-forward curve's last point and the 'robustness'
analysis's full-window vote_acc for this dt. Writes
robustness_time_profile_pct.csv and robustness_accuracy_over_pct_dt<X>ms.png.

--analysis picks which of the two to actually run: 'robustness', 'time-profile',
or both (default: 'robustness' alone, or both if --time-profile-dt-ms is given,
for backward compatibility). Running only one leaves the other's existing
output files on disk untouched — handy to add a time-profile pass without
re-touching an already-computed robustness_summary.csv, or vice versa.

Usage:
  python robustness_eval.py \
      --plain-experiment 26_09_01_16_133015_gesture_fwdPass_64_2_300K \
      --roi-parent       results_gesture/roi_matrix \
      [--roi-experiment ONE_MORE_DIR] \
      [--analysis robustness time-profile] \
      [--train] [--epochs N] [--model-name retrained-model.keras] \
      [--inter-frame-ms 1 10 50 100 200] [--window-ms 1000 0] \
      [--time-profile-dt-ms 100] [--out-dir robustness_out]
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


def _center_of_mass_seq(pos: np.ndarray) -> np.ndarray:
    """[T, H, W, 1] activation map -> [T, 2] (mean_y, mean_x) center of mass per
    frame — same computation as roigesture_coords in components/gesture_dataset.py."""
    A = pos[..., 0]  # [T, H, W]
    _, H, W = A.shape
    yy, xx = np.indices((H, W))
    tot = A.sum(axis=(1, 2))
    tot_safe = np.where(tot > 0, tot, 1.0)
    mean_y = (A * yy).sum(axis=(1, 2)) / tot_safe
    mean_x = (A * xx).sum(axis=(1, 2)) / tot_safe
    return np.stack([mean_y, mean_x], axis=1).astype(np.float32)


def _roi_frames_at_dt(dataset_path, frame_size, dt_us, window_us, keep_idx, coords: bool = False):
    """Yield ([T, fs, fs, 2], pos) float32 stacks for kept recordings.

    pos is [T, fs, fs, 1] for roigesture_matrix, or [T, 2] (center of mass,
    ``coords=True``) for roigesture_coords."""
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
        data, pos = data[:n], pos[:n]
        if coords:
            pos = _center_of_mass_seq(pos)  # [T, 2]
        yield data, pos


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


def _predict_sequences(model, frame_iter, labels, is_roi: bool):
    """Run the model once over every recording and return its raw per-frame
    predictions — the EXPENSIVE part of the time-profile analysis (the only
    part that actually needs the neural network). Kept separate from voting
    so a change to max_vote_frames or to --window-ms never re-runs inference:
    see _run_time_profile's raw-prediction cache (robustness_time_profile_raw.csv).

    Returns:
        list of (seq, y_true) per recording, seq = [T_i] int array of the
        per-frame predicted class (argmax of the logits).
    """
    sequences = []
    for item, y_true in zip(frame_iter, labels):
        if is_roi:
            data, pos = item
        else:
            data, pos = item, None
        logits = _infer_frames(model, data, pos, is_roi)         # [T, C]
        seq = logits.argmax(axis=1).astype(np.int64)             # [T]
        if len(seq) == 0:
            continue
        sequences.append((seq, int(y_true)))
    return sequences


def _vote_correctness(seq: np.ndarray, y_true: int, n_classes: int,
                       max_vote_frames: "int | None" = None) -> np.ndarray:
    """Running majority-vote correctness for one recording's predicted-class
    sequence, at every frame index k (0-based): True where the vote using
    frames seen so far (or only the last ``max_vote_frames`` of them, if
    given — a sliding window instead of accumulating since the start; e.g. 64
    to match a model trained on 64-frame sequences, so a long recording is
    always voted on the way the model actually saw data during training
    rather than an ever-larger, out-of-distribution history) equals y_true.

    This is the cheap, model-free part of the time-profile analysis — pure
    bookkeeping over an already-computed prediction sequence (see
    _predict_sequences), so re-running it with a different max_vote_frames
    costs nothing close to re-running the model.
    """
    T = len(seq)
    counts = np.zeros(n_classes, dtype=np.int64)
    correct = np.empty(T, dtype=bool)
    for k in range(T):
        counts[seq[k]] += 1
        if max_vote_frames is not None and k >= max_vote_frames:
            counts[seq[k - max_vote_frames]] -= 1  # slide the window: drop the oldest frame
        vote = int(counts.argmax())
        correct[k] = (vote == y_true)
    return correct


def _accuracy_over_time(sequences, n_classes: int = 11, max_vote_frames: "int | None" = None):
    """Running majority-vote accuracy at every elapsed-time checkpoint (after
    1, 2, 3, ... frames observed), for a FIXED inter-frame time, from
    already-computed prediction sequences (see _predict_sequences) — this
    function itself never touches the model.

    Answers "given dt=X ms fixed, how does accuracy evolve as more time/frames
    are observed?" — the companion of _stabilisation_frame (which asks *when*
    the vote stops oscillating, not whether it is correct).

    Each recording is used UNCROPPED, only up to its own actual length: once a
    recording runs out of frames it simply stops contributing to later
    checkpoints (no carry-forward of its last vote) — it "drops out" of the
    average rather than padding it. So the sample size shrinks at later
    checkpoints, down to whichever recording(s) ran longest.

    Args:
        sequences: list of (seq, y_true) per recording, from _predict_sequences.
        max_vote_frames: see _vote_correctness. None (default) keeps the
            original unbounded, since-the-beginning vote.

    Returns:
        (acc_curve, n_curve, per_sample_correct):
          - acc_curve, n_curve: both 1-D arrays as long as the longest
            recording (in frames): acc_curve[k] is the mean accuracy, across
            all recordings that still have a k-th frame, of the majority vote
            using their first k+1 frames (or last max_vote_frames, if set);
            n_curve[k] is how many that was.
          - per_sample_correct: the raw list of per-recording running-vote
            correctness arrays (one bool array of length T_i per recording),
            e.g. to build the duration-normalised curve (see
            _accuracy_over_pct_duration) without re-running inference.
    """
    per_sample_correct = [_vote_correctness(seq, y_true, n_classes, max_vote_frames)
                           for seq, y_true in sequences]

    if not per_sample_correct:
        return np.zeros(0), np.zeros(0, dtype=int), []

    max_len = max(len(a) for a in per_sample_correct)
    acc_curve = np.empty(max_len)
    n_curve = np.empty(max_len, dtype=int)
    for k in range(max_len):
        vals = [a[k] for a in per_sample_correct if len(a) > k]
        n_curve[k] = len(vals)
        acc_curve[k] = np.mean(vals) if vals else 0.0
    return acc_curve, n_curve, per_sample_correct


def _accuracy_over_pct_duration(per_sample_correct, n_steps: int = 100):
    """Accuracy vs % of EACH recording's OWN duration observed (checkpoints at
    1/n_steps, 2/n_steps, ..., 100%), instead of vs absolute elapsed time.

    Every recording maps onto every percentage checkpoint (its own nearest
    frame index), so — unlike _accuracy_over_time's absolute-time curve —
    n_samples is constant (every recording contributes throughout), and by
    construction the 100% point is EXACTLY "each recording's own full-length
    majority vote, averaged over all recordings" — the same quantity as the
    'robustness' analysis's "full" window vote_acc for the same dt.

    Args:
        per_sample_correct: from _accuracy_over_time — one bool array (running
            vote correctness) per recording, of that recording's own length.
        n_steps: number of percentage checkpoints (default 100 -> 1% steps).

    Returns:
        (pct_curve, n_samples) — pct_curve is a [n_steps] array, pct_curve[j]
        is the mean accuracy at (j+1)/n_steps of each recording's own length.
    """
    usable = [a for a in per_sample_correct if len(a) > 0]
    if not usable:
        return np.zeros(n_steps), 0
    pct_curve = np.empty(n_steps)
    for j in range(n_steps):
        frac = (j + 1) / n_steps
        vals = []
        for a in usable:
            T = len(a)
            k = min(max(int(round(frac * T)) - 1, 0), T - 1)
            vals.append(a[k])
        pct_curve[j] = np.mean(vals)
    return pct_curve, len(usable)


def _accuracy_over_time_carryforward(per_sample_correct):
    """Like the (drop-out) elapsed-time curve, but a recording that runs out
    of frames KEEPS contributing its last computed vote for every later
    checkpoint — "as if deployed continuously with no further evidence" —
    instead of dropping out of the average. Every recording therefore
    contributes to every checkpoint up to the longest recording's length, so
    n_samples is constant (= the total number of recordings), unlike the
    drop-out curve where it shrinks over time.

    Args:
        per_sample_correct: from _accuracy_over_time — one bool array (running
            vote correctness) per recording, of that recording's own length.

    Returns:
        (acc_curve, n_samples) — acc_curve is a 1-D array as long as the
        longest recording; n_samples is the (constant) recording count.
    """
    usable = [a for a in per_sample_correct if len(a) > 0]
    if not usable:
        return np.zeros(0), 0
    max_len = max(len(a) for a in usable)
    acc_curve = np.empty(max_len)
    for k in range(max_len):
        vals = [a[min(k, len(a) - 1)] for a in usable]  # carry the last value forward
        acc_curve[k] = np.mean(vals)
    return acc_curve, len(usable)


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
                        label=f"{kind}  — vote")
        ax.plot(x, [r["per_frame_acc"] for r in rows], "--x", alpha=0.5,
                color=line.get_color(), label=f"{kind} — per-frame")
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
            line, = ax.plot(x, p50, "-o", label=f"{kind}")
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


def _plot_time_profile(rows, out_path: Path, wtag: str, dt_ms: float, title_note: str = ""):
    """Accuracy vs elapsed observation time, at a FIXED inter-frame time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(dict.fromkeys(r["model"] for r in rows))
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in models:
        pts = sorted((r for r in rows if r["model"] == m), key=lambda r: r["elapsed_ms"])
        if not pts:
            continue
        kind = pts[0].get("kind", "")
        ax.plot([r["elapsed_ms"] for r in pts], [r["accuracy"] for r in pts], "-o",
                markersize=3, label=f"{kind}")
    ax.set_xlabel("Elapsed observation time (ms)", fontsize=14)
    ax.set_ylabel("Majority-vote accuracy so far", fontsize=14)
    ax.tick_params(labelsize=14)
    lim_x = min(max(r["elapsed_ms"] for r in rows), 5000000)
    ax.set_ylim(0.5, 1)
    ax.set_xlim(0, lim_x)
    ax.grid(True, alpha=0.3)
    suffix = f" — {title_note}" if title_note else ""
    ax.set_title(f"Accuracy vs elapsed time — fixed inter-frame time = {dt_ms:g} ms",fontsize=16)# ({wtag}){suffix}")
    ax.legend(fontsize=14, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _plot_time_profile_pct(rows, out_path: Path, dt_ms: float):
    """Accuracy vs % of each recording's OWN duration observed, at a FIXED
    inter-frame time. Every model's curve ends, at 100%, on the same number
    as its 'robustness'/full vote_acc for this dt (both average each
    recording's own full-length majority vote)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(dict.fromkeys(r["model"] for r in rows))
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in models:
        pts = sorted((r for r in rows if r["model"] == m), key=lambda r: r["pct_duration"])
        if not pts:
            continue
        kind = pts[0].get("kind", "")
        ax.plot([r["pct_duration"] for r in pts], [r["accuracy"] for r in pts], "-",
                label=f"{kind}")
    ax.set_xlabel("% of recording's own duration observed")
    ax.set_ylabel("Majority-vote accuracy so far")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Accuracy vs % of own duration observed — fixed inter-frame time = {dt_ms:g} ms\n"
                 f"(100% = same quantity as 'robustness' full-window vote_acc)")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")


_SUMM_INT_FIELDS = {"n_samples"}
_SUMM_FLOAT_FIELDS = {"inter_frame_ms", "vote_acc", "per_frame_acc", "mean_n_frames",
                       "stab_frame_p25", "stab_frame_p50", "stab_frame_p75",
                       "stab_ms_p25", "stab_ms_p50", "stab_ms_p75"}
_SAMPLE_INT_FIELDS = {"sample", "true", "pred_vote", "correct", "n_frames", "stab_frame", "stable_label"}
_SAMPLE_FLOAT_FIELDS = {"inter_frame_ms", "per_frame_acc", "stab_ms"}
_TIMEPROF_INT_FIELDS = {"n_samples", "max_vote_frames"}
_TIMEPROF_FLOAT_FIELDS = {"dt_ms", "elapsed_ms", "accuracy"}
_TIMEPROF_PCT_INT_FIELDS = {"n_samples", "max_vote_frames"}
_TIMEPROF_PCT_FLOAT_FIELDS = {"dt_ms", "pct_duration", "accuracy"}
_TIMEPROF_CF_INT_FIELDS = {"n_samples", "max_vote_frames"}
_TIMEPROF_CF_FLOAT_FIELDS = {"dt_ms", "elapsed_ms", "accuracy"}
_RAWPRED_INT_FIELDS = {"sample", "true", "frame", "pred"}
_RAWPRED_FLOAT_FIELDS = {"dt_ms"}


def _norm_mvf(v) -> "int | None":
    """Normalise a max_vote_frames value (possibly read back from CSV as ''
    or missing) to None (unbounded) or an int, for cache-key comparisons."""
    return None if v in (None, "") else int(v)


def _coerce_row(row: dict, int_fields, float_fields) -> dict:
    out = dict(row)
    for k in int_fields:
        if out.get(k) not in (None, ""):
            out[k] = int(float(out[k]))
    for k in float_fields:
        if out.get(k) not in (None, ""):
            out[k] = float(out[k])
    return out


def _load_existing(out_dir: Path):
    """Load a previous run's CSVs, if any, so unchanged experiments can be skipped."""
    summ, per_sample, time_profile, time_profile_pct, time_profile_cf = [], [], [], [], []
    summ_path = out_dir / "robustness_summary.csv"
    per_path = out_dir / "robustness_per_sample.csv"
    tp_path = out_dir / "robustness_time_profile.csv"
    tpp_path = out_dir / "robustness_time_profile_pct.csv"
    tpcf_path = out_dir / "robustness_time_profile_carryforward.csv"
    if summ_path.exists():
        with open(summ_path, newline="") as f:
            summ = [_coerce_row(r, _SUMM_INT_FIELDS, _SUMM_FLOAT_FIELDS) for r in csv.DictReader(f)]
    if per_path.exists():
        with open(per_path, newline="") as f:
            per_sample = [_coerce_row(r, _SAMPLE_INT_FIELDS, _SAMPLE_FLOAT_FIELDS) for r in csv.DictReader(f)]
    if tp_path.exists():
        with open(tp_path, newline="") as f:
            time_profile = [_coerce_row(r, _TIMEPROF_INT_FIELDS, _TIMEPROF_FLOAT_FIELDS) for r in csv.DictReader(f)]
    if tpp_path.exists():
        with open(tpp_path, newline="") as f:
            time_profile_pct = [_coerce_row(r, _TIMEPROF_PCT_INT_FIELDS, _TIMEPROF_PCT_FLOAT_FIELDS)
                                 for r in csv.DictReader(f)]
    if tpcf_path.exists():
        with open(tpcf_path, newline="") as f:
            time_profile_cf = [_coerce_row(r, _TIMEPROF_CF_INT_FIELDS, _TIMEPROF_CF_FLOAT_FIELDS)
                                for r in csv.DictReader(f)]
    raw_path = out_dir / "robustness_time_profile_raw.csv"
    raw_predictions = []
    if raw_path.exists():
        with open(raw_path, newline="") as f:
            raw_predictions = [_coerce_row(r, _RAWPRED_INT_FIELDS, _RAWPRED_FLOAT_FIELDS)
                                for r in csv.DictReader(f)]
    return summ, per_sample, time_profile, time_profile_pct, time_profile_cf, raw_predictions


def _already_computed(existing_summ, kind: str, model: str, windows, inter_frame_ms) -> bool:
    """True if every requested (window, inter-frame) combo for this (kind, model)
    is already present in the existing summary with at least one sample."""
    want_windows = {"full" if w is None else f"{int(round(w / 1000))}ms" for w in windows}
    want_dt = set(inter_frame_ms)
    have = {
        (s["window"], s["inter_frame_ms"])
        for s in existing_summ
        if s.get("kind") == kind and s.get("model") == model and s.get("n_samples", 0)
    }
    return all((w, dt) in have for w in want_windows for dt in want_dt)


def _time_profile_already_computed(existing_tp, kind: str, model: str, window_us, dt_ms: float,
                                    max_vote_frames: "int | None" = None) -> bool:
    """True if this (kind, model, window, dt, max_vote_frames) already has time-profile rows."""
    wtag = "full" if window_us is None else f"{int(round(window_us / 1000))}ms"
    return any(
        r.get("kind") == kind and r.get("model") == model
        and r.get("window") == wtag and r.get("dt_ms") == dt_ms
        and _norm_mvf(r.get("max_vote_frames")) == max_vote_frames
        for r in existing_tp
    )


def _time_profile_pct_already_computed(existing_pct, kind: str, model: str, dt_ms: float,
                                        max_vote_frames: "int | None" = None) -> bool:
    """True if this (kind, model, dt, max_vote_frames) already has %-duration time-profile rows."""
    return any(
        r.get("kind") == kind and r.get("model") == model and r.get("dt_ms") == dt_ms
        and _norm_mvf(r.get("max_vote_frames")) == max_vote_frames
        for r in existing_pct
    )


def _time_profile_cf_already_computed(existing_cf, kind: str, model: str, window_us, dt_ms: float,
                                       max_vote_frames: "int | None" = None) -> bool:
    """True if this (kind, model, window, dt, max_vote_frames) already has carry-forward rows."""
    wtag = "full" if window_us is None else f"{int(round(window_us / 1000))}ms"
    return any(
        r.get("kind") == kind and r.get("model") == model
        and r.get("window") == wtag and r.get("dt_ms") == dt_ms
        and _norm_mvf(r.get("max_vote_frames")) == max_vote_frames
        for r in existing_cf
    )


def _raw_predictions_already_computed(existing_raw, kind: str, model: str, dt_ms: float) -> bool:
    """True if the raw per-frame model predictions for this (kind, model, dt) are
    already cached — independent of window/max_vote_frames, since raw predictions
    never depend on either (only the cheap voting step does)."""
    return any(r.get("kind") == kind and r.get("model") == model and r.get("dt_ms") == dt_ms
               for r in existing_raw)


def _reconstruct_sequences(existing_raw, kind: str, model: str, dt_ms: float):
    """Rebuild the list[(seq, y_true)] that _predict_sequences would have produced,
    from cached raw-prediction rows, so time-profile re-parametrization (a new
    max_vote_frames or --window-ms) never has to re-run the model."""
    by_sample = {}
    for r in existing_raw:
        if r.get("kind") != kind or r.get("model") != model or r.get("dt_ms") != dt_ms:
            continue
        by_sample.setdefault(r["sample"], []).append((r["frame"], r["pred"], r["true"]))
    sequences = []
    for sample in sorted(by_sample):
        frames = sorted(by_sample[sample], key=lambda t: t[0])
        seq = np.array([pred for _, pred, _ in frames], dtype=np.int64)
        y_true = frames[0][2]
        sequences.append((seq, y_true))
    return sequences


def _raw_rows_from_sequences(sequences, kind: str, model: str, dt_ms: float):
    """The inverse of _reconstruct_sequences: flatten freshly computed sequences
    into cacheable per-frame rows."""
    rows = []
    for sample, (seq, y_true) in enumerate(sequences):
        for frame, pred in enumerate(seq):
            rows.append({
                "kind": kind, "model": model, "dt_ms": dt_ms,
                "sample": sample, "true": int(y_true), "frame": frame, "pred": int(pred),
            })
    return rows


def _write_summary_csvs(summ, per_sample, out_dir: Path):
    """robustness_summary.csv + robustness_per_sample.csv (the 'robustness' analysis)."""
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


def _write_time_profile_csv(time_profile, out_dir: Path):
    """robustness_time_profile.csv (the 'time-profile' analysis)."""
    if not time_profile:
        return
    tfields = ["kind", "model", "window", "dt_ms", "elapsed_ms", "accuracy", "n_samples", "max_vote_frames"]
    with open(out_dir / "robustness_time_profile.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=tfields)
        w.writeheader()
        for r in time_profile:
            w.writerow({k: r.get(k, "") for k in tfields})
    print(f"  wrote {out_dir/'robustness_time_profile.csv'}")


def _write_time_profile_pct_csv(time_profile_pct, out_dir: Path):
    """robustness_time_profile_pct.csv (accuracy vs % of each recording's own
    duration observed — the companion of robustness_time_profile.csv whose
    100% point matches the 'robustness' analysis's full-window vote_acc)."""
    if not time_profile_pct:
        return
    tfields = ["kind", "model", "dt_ms", "pct_duration", "accuracy", "n_samples", "max_vote_frames"]
    with open(out_dir / "robustness_time_profile_pct.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=tfields)
        w.writeheader()
        for r in time_profile_pct:
            w.writerow({k: r.get(k, "") for k in tfields})
    print(f"  wrote {out_dir/'robustness_time_profile_pct.csv'}")


def _write_time_profile_carryforward_csv(time_profile_cf, out_dir: Path):
    """robustness_time_profile_carryforward.csv — same axis (elapsed_ms) as
    robustness_time_profile.csv, but a finished recording carries its last
    vote forward instead of dropping out, so n_samples stays constant."""
    if not time_profile_cf:
        return
    tfields = ["kind", "model", "window", "dt_ms", "elapsed_ms", "accuracy", "n_samples", "max_vote_frames"]
    with open(out_dir / "robustness_time_profile_carryforward.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=tfields)
        w.writeheader()
        for r in time_profile_cf:
            w.writerow({k: r.get(k, "") for k in tfields})
    print(f"  wrote {out_dir/'robustness_time_profile_carryforward.csv'}")


def _write_raw_predictions_csv(raw_predictions, out_dir: Path):
    """robustness_time_profile_raw.csv — the expensive part of the time-profile
    analysis (raw per-frame model predictions per recording, at a given
    (kind, model, dt_ms)), cached independently of window/max_vote_frames so
    re-parametrizing the vote never re-runs the model (see _reconstruct_sequences)."""
    if not raw_predictions:
        return
    rfields = ["kind", "model", "dt_ms", "sample", "true", "frame", "pred"]
    with open(out_dir / "robustness_time_profile_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rfields)
        w.writeheader()
        for r in raw_predictions:
            w.writerow({k: r.get(k, "") for k in rfields})
    print(f"  wrote {out_dir/'robustness_time_profile_raw.csv'}")


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
        dataset_path = "../rois_fs_32_16/datasets/"
        if not Path(dataset_path).exists():
            print(f"[{label}] ROI dataset '{dataset_path}' not found — skipping.")
            return [], []
        frame_size = _roi_frame_size(experiment)
        # roigesture_coords -> pos collapsed to its [T, 2] center of mass;
        # roigesture_matrix (or anything else) -> pos kept as a [T, fs, fs, 1] map.
        coords = "coords" in str(getattr(cfg, "dataset", "")).lower()
        print(f"[{label}] ROI variant: {'coords (center of mass)' if coords else 'matrix (position map)'}")
    else:
        dataset_path, frame_size, coords = "./data", None, False

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
                it = _roi_frames_at_dt(dataset_path, frame_size, dt_us, window_us, keep, coords=coords)
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


def _run_time_profile(experiment: Path, kind: str, is_roi: bool, args, windows,
                       existing_tp, existing_pct, existing_cf, existing_raw):
    """Accuracy-vs-observation analysis at a FIXED inter-frame time
    (``args.time_profile_dt_ms``), computed in THREE ways from the same
    inference pass:

      - vs absolute elapsed time, drop-out (robustness_time_profile.csv): a
        recording drops out of the average once its own frames run out (no
        carry-forward), so later checkpoints average over fewer,
        longer-lasting recordings.
      - vs absolute elapsed time, carry-forward
        (robustness_time_profile_carryforward.csv): same x-axis, but a
        finished recording keeps contributing its LAST vote for every later
        checkpoint instead of dropping out, so n_samples stays constant (see
        _accuracy_over_time_carryforward).
      - vs % of each recording's OWN duration (robustness_time_profile_pct.csv):
        every recording contributes at every percentage checkpoint, so
        n_samples is constant, and the 100% point equals the 'robustness'
        analysis's full-window vote_acc for this dt (see
        _accuracy_over_pct_duration).

    The underlying computation always uses each recording UNCROPPED (no
    population filtering, no discard by duration) — ``windows`` is used ONLY
    to also emit a TRUNCATED copy of the two elapsed-time curves (drop-out and
    carry-forward) for each finite value, i.e. the same curve cut at
    elapsed_ms <= window, not a different (cropped/filtered) computation. This
    is why, unlike the 'robustness' analysis, a windowed curve here is
    guaranteed to be an exact prefix of the "full" one. The %-duration curve
    is unaffected by ``windows`` (it is already bounded to 0-100%).

    Cached independently of the main sweep, in two layers: the EXPENSIVE raw
    per-frame model predictions are cached per (kind, model, dt) alone, in
    robustness_time_profile_raw.csv, since they never depend on window or
    max_vote_frames — only the CHEAP voting/aggregation step below does. So
    changing --time-profile-max-vote-frames or adding a new --window-ms value
    reuses the cached predictions and only re-runs the cheap part; the model
    is only re-invoked the first time a given (kind, model, dt) is seen. The
    three CSV outputs are still skipped individually when every requested
    window tag already has rows for the current (kind, model, dt,
    max_vote_frames) — so upgrading from an older run still fills in just
    what's missing.
    """
    from exp_config import set_active_config, load_cfg

    dt_ms = args.time_profile_dt_ms
    max_vote_frames = getattr(args, "time_profile_max_vote_frames", None)
    config_path = experiment / "config.yaml"
    if not config_path.exists():
        print(f"[{experiment.name}] no config.yaml — skipping time-profile")
        return [], [], [], []
    set_active_config(config_path)
    cfg = load_cfg(force=True)
    label = experiment.name
    n_classes = cfg_n_classes(cfg)

    # "full" (untruncated) always included, plus one truncation per finite window.
    window_us_list = [None] + sorted({w for w in windows if w is not None})

    have_abs = all(_time_profile_already_computed(existing_tp, kind, label, w, dt_ms, max_vote_frames)
                   for w in window_us_list)
    have_cf = all(_time_profile_cf_already_computed(existing_cf, kind, label, w, dt_ms, max_vote_frames)
                  for w in window_us_list)
    have_pct = _time_profile_pct_already_computed(existing_pct, kind, label, dt_ms, max_vote_frames)
    if have_abs and have_pct and have_cf:
        print(f"[{label}] time-profile (elapsed drop-out + carry-forward + % duration) already "
              f"computed for dt={dt_ms:g}ms, max_vote_frames={max_vote_frames} and all requested "
              f"windows — skipping recompute")
        return [], [], [], []

    have_raw = _raw_predictions_already_computed(existing_raw, kind, label, dt_ms)
    new_raw_rows = []
    if have_raw:
        sequences = _reconstruct_sequences(existing_raw, kind, label, dt_ms)
        print(f"[{label}] reusing cached raw predictions for dt={dt_ms:g}ms "
              f"({len(sequences)} recordings) — skipping model inference")
    else:
        if is_roi:
            dataset_path = "../rois_fs_32_16/datasets/"
            if not Path(dataset_path).exists():
                print(f"[{label}] ROI dataset '{dataset_path}' not found — skipping time-profile.")
                return [], [], [], []
            frame_size = _roi_frame_size(experiment)
            coords = "coords" in str(getattr(cfg, "dataset", "")).lower()
        else:
            dataset_path, frame_size, coords = "./data", None, False

        if is_roi:
            keep, labels = _keep_indices_roi(dataset_path, frame_size, None)
        else:
            keep, labels = _keep_indices_plain(dataset_path, None)
        if not keep:
            return [], [], [], []

        model = _resolve_model(experiment, args)
        dt_us = int(dt_ms * 1000)
        print(f"[{label}] time-profile (uncropped, own duration) dt={dt_ms:g}ms "
              f"over {len(keep)} recordings — running inference...")
        if is_roi:
            it = _roi_frames_at_dt(dataset_path, frame_size, dt_us, None, keep, coords=coords)
        else:
            it = _plain_frames_at_dt(dataset_path, dt_us, None, keep)
        sequences = _predict_sequences(model, it, labels, is_roi)
        new_raw_rows = _raw_rows_from_sequences(sequences, kind, label, dt_ms)

    vote_note = f", max_vote_frames={max_vote_frames}" if max_vote_frames is not None else ""
    print(f"[{label}] voting{vote_note} over {len(sequences)} recordings...")
    acc_curve, n_curve, per_sample_correct = _accuracy_over_time(
        sequences, n_classes=n_classes, max_vote_frames=max_vote_frames
    )
    if len(acc_curve):
        print(f"    drop-out:      n@{dt_ms:g}ms={n_curve[0]}  acc={acc_curve[0]:.4f}   "
              f"n@{len(acc_curve)*dt_ms:g}ms={n_curve[-1]}  acc={acc_curve[-1]:.4f}")

    cf_curve, n_cf = (np.zeros(0), 0)
    if not have_cf:
        cf_curve, n_cf = _accuracy_over_time_carryforward(per_sample_correct)
        if len(cf_curve):
            print(f"    carry-forward: n={n_cf} (constant)  acc@{dt_ms:g}ms={cf_curve[0]:.4f}   "
                  f"acc@{len(cf_curve)*dt_ms:g}ms={cf_curve[-1]:.4f}")

    def _rows_for(curve, n_of_k, is_constant_n: bool):
        rows = []
        for window_us in window_us_list:
            wtag = "full" if window_us is None else f"{int(round(window_us / 1000))}ms"
            limit_ms = None if window_us is None else window_us / 1000
            for k in range(len(curve)):
                elapsed = (k + 1) * dt_ms
                if limit_ms is not None and elapsed > limit_ms:
                    break
                rows.append({
                    "kind": kind, "model": label, "window": wtag, "dt_ms": dt_ms,
                    "elapsed_ms": elapsed, "accuracy": float(curve[k]),
                    "n_samples": int(n_of_k) if is_constant_n else int(n_of_k[k]),
                    "max_vote_frames": max_vote_frames,
                })
        return rows

    out_rows = [] if have_abs else _rows_for(acc_curve, n_curve, is_constant_n=False)
    cf_rows = [] if have_cf else _rows_for(cf_curve, n_cf, is_constant_n=True)

    pct_rows = []
    if not have_pct:
        n_steps = getattr(args, "time_profile_pct_steps", 100)
        pct_curve, n_pct = _accuracy_over_pct_duration(per_sample_correct, n_steps=n_steps)
        pct_rows = [
            {
                "kind": kind, "model": label, "dt_ms": dt_ms,
                "pct_duration": (j + 1) / n_steps * 100, "accuracy": float(pct_curve[j]), "n_samples": n_pct,
                "max_vote_frames": max_vote_frames,
            }
            for j in range(len(pct_curve))
        ]
        if len(pct_curve):
            print(f"    %duration:     n={n_pct}  acc@1%={pct_curve[0]:.4f}  acc@100%={pct_curve[-1]:.4f}  "
                  f"(<-> 'robustness' full vote_acc)")
    return out_rows, pct_rows, cf_rows, new_raw_rows


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


def _roi_kind_label(experiment: Path) -> str:
    """"ROI-coords" / "ROI-matrix" (from config.yaml's dataset field), or the
    generic "ROI" if it can't be determined without fully loading the config."""
    try:
        import yaml
        with open(experiment / "config.yaml") as f:
            raw = yaml.safe_load(f) or {}
        name = str(raw.get("dataset", "")).lower()
        if "coords" in name:
            return "ROI-coords"
        if "matrix" in name:
            return "ROI-matrix"
    except Exception:
        pass
    return "ROI"


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
                dir_resolved = d.resolve()
                print(f"[plan] found ROI experiment: {dir_resolved}")
                if "fwdPass" not in dir_resolved.name:
                    print(f"[plan] skipping {dir_resolved} (non forward-pass experiment)")
                    continue
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
                   help="Windows to evaluate, in ms. 0 = no limit. Default: 1000 0 (both the 1 s "
                        "window and the unlimited case). For 'robustness' each value CROPS the "
                        "recordings and DISCARDS ones shorter than it (a different population per "
                        "window). For 'time-profile' each finite value only TRUNCATES the output "
                        "(elapsed_ms <= window) of the already-uncropped drop-out/carry-forward "
                        "curves — same population and values as 'full', just cut short; it never "
                        "affects the %%-duration curve, which is unaffected by --window-ms.")
    p.add_argument("--time-profile-dt-ms", type=float, default=None, dest="time_profile_dt_ms",
                   help="Fix the inter-frame time to this value (ms) and measure accuracy vs "
                        "ELAPSED OBSERVATION TIME (running majority vote after 1, 2, 3, ... "
                        "frames), instead of accuracy vs inter-frame time. Always uses each "
                        "recording UNCROPPED and only up to its own actual duration — no --window-ms "
                        "crop, no discard by length, no carry-forward past a recording's own end "
                        "(it just drops out of the average once it runs out of frames). Required "
                        "for --analysis time-profile. Writes robustness_time_profile.csv and "
                        "robustness_accuracy_over_time_dt<dt>ms_full.png, PLUS the %%-duration "
                        "companion below.")
    p.add_argument("--time-profile-pct-steps", type=int, default=100, dest="time_profile_pct_steps",
                   help="Resolution (number of checkpoints from 1%% to 100%%) of the %%-of-own-"
                        "duration curve computed alongside --time-profile-dt-ms (default: 100, "
                        "i.e. 1%% steps). Every recording contributes at every checkpoint (mapped "
                        "to its own nearest frame), so unlike the elapsed-time curve, n_samples is "
                        "constant and the 100%% point equals the 'robustness' full-window vote_acc "
                        "for this dt. Writes robustness_time_profile_pct.csv and "
                        "robustness_accuracy_over_pct_dt<dt>ms.png.")
    p.add_argument("--time-profile-max-vote-frames", type=int, default=None, dest="time_profile_max_vote_frames",
                   help="Cap the majority vote used by ALL THREE --time-profile-dt-ms curves "
                        "(drop-out, carry-forward, %%-duration) to a SLIDING WINDOW of at most this "
                        "many of the most recent frames, instead of every frame since the start of "
                        "the recording. E.g. 64 to match a model trained on 64-frame sequences, so "
                        "long recordings are always voted on the way the model actually saw data "
                        "during training rather than accumulating an ever-larger, "
                        "out-of-distribution history. Default: None (unbounded, since-the-start "
                        "vote — the original behaviour). Part of the cache key, so switching this "
                        "does not reuse/collide with results computed at a different value, and "
                        "output filenames get a '_vote<N>' suffix when set.")
    p.add_argument("--analysis", type=str, nargs="+", default=None,
                   choices=["robustness", "time-profile"],
                   help="Which analysis(es) to run: 'robustness' (accuracy + stabilisation vs "
                        "inter-frame time; robustness_summary/per_sample.csv + their plots), "
                        "'time-profile' (accuracy vs elapsed time at a fixed --time-profile-dt-ms; "
                        "robustness_time_profile.csv + its plots), or both. Running only one "
                        "leaves the other's existing output files untouched. Default: 'robustness' "
                        "alone, or both if --time-profile-dt-ms is given (back-compat).")
    p.add_argument("--out-dir", type=str, default="robustness_out")
    args = p.parse_args()

    roi_dirs = _roi_experiment_dirs(args)
    if not args.plain_experiment and not roi_dirs:
        p.error("give --plain-experiment and/or --roi-experiment / --roi-parent")

    if args.analysis is None:
        # Back-compat default: robustness always; time-profile too if a dt was given.
        analyses = {"robustness"} | ({"time-profile"} if args.time_profile_dt_ms is not None else set())
    else:
        analyses = set(args.analysis)
    if "time-profile" in analyses and args.time_profile_dt_ms is None:
        p.error("--analysis time-profile requires --time-profile-dt-ms")

    windows = [None if w == 0 else int(round(w * 1000)) for w in args.window_ms]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_summ, existing_per_sample, existing_tp, existing_pct, existing_cf, existing_raw = _load_existing(out_dir)

    targets = []
    if args.plain_experiment:
        targets.append((Path(args.plain_experiment).resolve(), "non-ROI", False))
    for d in roi_dirs:
        targets.append((d, _roi_kind_label(d), True))

    print(f"[plan] analysis={sorted(analyses)}, {len(targets)} experiment(s), "
          f"windows={['full' if w is None else f'{w//1000}ms' for w in windows]}, "
          f"inter-frame ms={args.inter_frame_ms}")

    all_summ, all_per_sample = [], []
    if "robustness" in analyses:
        requested_windows = {"full" if w is None else f"{int(round(w / 1000))}ms" for w in windows}
        requested_dt = set(args.inter_frame_ms)
        for experiment, kind, is_roi in targets:
            model_name = experiment.name
            if _already_computed(existing_summ, kind, model_name, windows, args.inter_frame_ms):
                print(f"\n[{kind}] {model_name}: already computed in {out_dir}/robustness_summary.csv "
                      f"for the requested windows/inter-frame times — skipping recompute.")
                all_summ += [s for s in existing_summ
                             if s.get("kind") == kind and s.get("model") == model_name
                             and s.get("window") in requested_windows and s.get("inter_frame_ms") in requested_dt]
                all_per_sample += [r for r in existing_per_sample
                                    if r.get("kind") == kind and r.get("model") == model_name
                                    and r.get("window") in requested_windows and r.get("inter_frame_ms") in requested_dt]
                continue

            print(f"\n{'='*70}\n{kind}: {experiment}\n{'='*70}")
            try:
                summ, per_sample = _run_experiment(experiment, kind, is_roi, args, windows)
            except Exception as exc:
                print(f"[ERROR] {experiment}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            all_summ.extend(summ)
            all_per_sample.extend(per_sample)
    else:
        print("[plan] --analysis does not include 'robustness' — leaving robustness_summary/"
              "per_sample.csv and their plots untouched.")

    all_time_profile, all_time_profile_pct, all_time_profile_cf, all_raw = [], [], [], []
    if "time-profile" in analyses:
        for experiment, kind, is_roi in targets:
            try:
                rows, pct_rows, cf_rows, raw_rows = _run_time_profile(
                    experiment, kind, is_roi, args, windows,
                    existing_tp, existing_pct, existing_cf, existing_raw
                )
                all_time_profile.extend(rows)
                all_time_profile_pct.extend(pct_rows)
                all_time_profile_cf.extend(cf_rows)
                all_raw.extend(raw_rows)
            except Exception as exc:
                print(f"[ERROR] time-profile {experiment}: {type(exc).__name__}: {exc}", file=sys.stderr)
        # keep cached raw predictions for (kind, model, dt) combos we didn't just recompute
        new_raw_keys = {(r["kind"], r["model"], r["dt_ms"]) for r in all_raw}
        all_raw += [r for r in existing_raw
                    if (r.get("kind"), r.get("model"), r.get("dt_ms")) not in new_raw_keys]
        # keep whatever from a previous run is still relevant (other dt/window/vote-window combos)
        new_keys = {(r["kind"], r["model"], r["window"], r["dt_ms"], _norm_mvf(r.get("max_vote_frames")))
                    for r in all_time_profile}
        all_time_profile += [r for r in existing_tp
                              if (r.get("kind"), r.get("model"), r.get("window"), r.get("dt_ms"),
                                  _norm_mvf(r.get("max_vote_frames"))) not in new_keys]
        new_pct_keys = {(r["kind"], r["model"], r["dt_ms"], _norm_mvf(r.get("max_vote_frames")))
                        for r in all_time_profile_pct}
        all_time_profile_pct += [r for r in existing_pct
                                  if (r.get("kind"), r.get("model"), r.get("dt_ms"),
                                      _norm_mvf(r.get("max_vote_frames"))) not in new_pct_keys]
        new_cf_keys = {(r["kind"], r["model"], r["window"], r["dt_ms"], _norm_mvf(r.get("max_vote_frames")))
                       for r in all_time_profile_cf}
        all_time_profile_cf += [r for r in existing_cf
                                 if (r.get("kind"), r.get("model"), r.get("window"), r.get("dt_ms"),
                                     _norm_mvf(r.get("max_vote_frames"))) not in new_cf_keys]
    else:
        print("[plan] --analysis does not include 'time-profile' — leaving "
              "robustness_time_profile*.csv and its plots untouched.")

    if "robustness" in analyses and not all_summ:
        print("[WARNING] 'robustness' analysis requested but produced no results.")
    if "time-profile" in analyses and not all_time_profile and not all_time_profile_pct and not all_time_profile_cf:
        print("[WARNING] 'time-profile' analysis requested but produced no results.")
    if not all_summ and not all_time_profile and not all_time_profile_pct and not all_time_profile_cf:
        print("[ERROR] nothing evaluated.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*70}\nWriting outputs to {out_dir}/\n{'='*70}")

    if "robustness" in analyses:
        _write_summary_csvs(all_summ, all_per_sample, out_dir)
        for wtag in dict.fromkeys(s["window"] for s in all_summ):
            sub = [s for s in all_summ if s["window"] == wtag]
            _plot_accuracy(sub, out_dir / f"robustness_accuracy_{wtag}.png", wtag)
            _plot_stabilisation(sub, out_dir / f"robustness_stabilisation_{wtag}.png", wtag)

    def _vote_suffix(mvf) -> str:
        mvf = _norm_mvf(mvf)
        return f"_vote{mvf}" if mvf is not None else ""

    if "time-profile" in analyses:
        _write_time_profile_csv(all_time_profile, out_dir)
        for dt_ms in dict.fromkeys(r["dt_ms"] for r in all_time_profile):
            for mvf in dict.fromkeys(_norm_mvf(r.get("max_vote_frames")) for r in all_time_profile if r["dt_ms"] == dt_ms):
                for wtag in dict.fromkeys(r["window"] for r in all_time_profile
                                          if r["dt_ms"] == dt_ms and _norm_mvf(r.get("max_vote_frames")) == mvf):
                    sub = [r for r in all_time_profile if r["dt_ms"] == dt_ms and r["window"] == wtag
                           and _norm_mvf(r.get("max_vote_frames")) == mvf]
                    _plot_time_profile(
                        sub,
                        out_dir / f"robustness_accuracy_over_time_dt{dt_ms:g}ms_{wtag}{_vote_suffix(mvf)}.png",
                        wtag, dt_ms,
                        #title_note="drop-out",
                    )

        _write_time_profile_carryforward_csv(all_time_profile_cf, out_dir)
        for dt_ms in dict.fromkeys(r["dt_ms"] for r in all_time_profile_cf):
            for mvf in dict.fromkeys(_norm_mvf(r.get("max_vote_frames")) for r in all_time_profile_cf if r["dt_ms"] == dt_ms):
                for wtag in dict.fromkeys(r["window"] for r in all_time_profile_cf
                                          if r["dt_ms"] == dt_ms and _norm_mvf(r.get("max_vote_frames")) == mvf):
                    sub = [r for r in all_time_profile_cf if r["dt_ms"] == dt_ms and r["window"] == wtag
                           and _norm_mvf(r.get("max_vote_frames")) == mvf]
                    _plot_time_profile(
                        sub,
                        out_dir / (f"robustness_accuracy_over_time_dt{dt_ms:g}ms_{wtag}"
                                   f"{_vote_suffix(mvf)}_carryforward.png"),
                        wtag, dt_ms, title_note="carry-forward",
                    )

        _write_time_profile_pct_csv(all_time_profile_pct, out_dir)
        for dt_ms in dict.fromkeys(r["dt_ms"] for r in all_time_profile_pct):
            for mvf in dict.fromkeys(_norm_mvf(r.get("max_vote_frames")) for r in all_time_profile_pct if r["dt_ms"] == dt_ms):
                sub = [r for r in all_time_profile_pct if r["dt_ms"] == dt_ms
                       and _norm_mvf(r.get("max_vote_frames")) == mvf]
                _plot_time_profile_pct(
                    sub, out_dir / f"robustness_accuracy_over_pct_dt{dt_ms:g}ms{_vote_suffix(mvf)}.png", dt_ms
                )

        _write_raw_predictions_csv(all_raw, out_dir)


if __name__ == "__main__":
    main()



