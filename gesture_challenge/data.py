"""
DVSGesture ROI challenge - dataset loader (fwdPass only).

Standalone module (numpy + tonic, no deep-learning framework): it reproduces the
fwdPass preprocessing used by the Symbolic DNN-Tuner for the ROI version of
DVSGesture, so results are comparable with the networks found by the tuner.

For every recording the dataset gives two pieces of information:

  roi      what happens inside a 32x32 region of interest (ROI) that follows the
           gesture: event frames [T, 32, 32, 2] (2 = event polarities)
  pos      where the ROI is on the 128x128 sensor, in two forms:
             - pos_map:    the ROI position framed like the roi frames and
                           downsampled to 32x32: [T, 32, 32, 1]
             - pos_events: the raw position events of the recording
                           (structured array with fields x, y, s, p, t; x, y =
                           ROI centre in 128x128 sensor coordinates, t in microseconds)

Labels y are integers in [0, 10] (11 classes), one per recording.

fwdPass: every recording is a sequence of T frames, the network classifies ONE
frame at a time and the recording prediction is the average of the per-frame
probabilities (see evaluate.py). How to combine roi and pos is up to you.

Framing (same as the tuner):
  - train: "frames"  -> every recording split into `frames` equal time bins (T = frames)
  - test:  "delta_t" -> fixed time window of `delta_t` us; every recording is cut to
                        its first 1.5 s, so T = 1_500_000 // delta_t
Both can be changed with the `framing` argument of load_split.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import tonic.transforms as transforms
from tonic.dataset import Dataset
from tonic.datasets import DVSGesture


N_CLASSES = 11
CLASS_NAMES = [
    "hand_clapping", "right_hand_wave", "left_hand_wave", "right_arm_clockwise",
    "right_arm_counter_clockwise", "left_arm_clockwise", "left_arm_counter_clockwise",
    "arm_roll", "air_drums", "air_guitar", "other_gestures",
]

# delta_t framing: every recording is cut to its first MAX_DURATION_US microseconds
# (every DVSGesture recording lasts at least ~1.75 s).
MAX_DURATION_US = 1_500_000

# Defaults used by the tuner experiments.
DEFAULT_FRAMES = 64
DEFAULT_DELTA_T = 200_000
DEFAULT_ROOT = "./rois_and_coordinates/datasets"
DEFAULT_CACHE_DIR = "./cache/challenge"
ROI_SIZE = 32
SENSOR_SIZE = 128


def center_roi_positions(position: np.ndarray) -> np.ndarray:
    """The position files store the top-left corner of the ROI: shift to the ROI
    centre (+16 per axis), clipped to the sensor pixels 0..127."""
    position = position.copy()
    for axis in ("x", "y"):
        position[axis] = np.clip(position[axis] + ROI_SIZE // 2, 0, SENSOR_SIZE - 1)
    return position


def n_window_frames(delta_t: int) -> int:
    """Number of delta_t frames kept per recording (MAX_DURATION_US // delta_t)."""
    n = MAX_DURATION_US // int(delta_t)
    if n < 1:
        raise ValueError(f"delta_t={delta_t} us is longer than the {MAX_DURATION_US} us window")
    return n


# ------------------------------- Dataset -------------------------------------

class DVSGestureROI(DVSGesture):
    """DVSGesture variant that returns the ROI events and the ROI position events.

    Expects the pre-extracted ROI dataset at <save_to>/DVSGestureROI_fs32/{train,test}/
    with, per recording, <label>.npy (ROI events) and <label>_positions.npy.
    """

    dtype = np.dtype([("x", np.int16), ("y", np.int16), ("p", bool), ("t", np.int64)])
    ordering = dtype.names

    def __init__(
        self,
        save_to: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        position_transform: Optional[Callable] = None,
    ):
        Dataset.__init__(self, save_to, transform=transform)
        self.location_on_system = self.location_on_system + f"_fs{ROI_SIZE}"
        self.position_transform = position_transform
        self.train = train
        self.folder_name = "train" if train else "test"

        self.position = []
        file_path = os.path.join(self.location_on_system, self.folder_name)
        if not os.path.isdir(file_path):
            raise FileNotFoundError(f"ROI dataset not found: {file_path}")
        for path, dirs, files in os.walk(file_path):
            dirs.sort()
            for file in sorted(files):
                if file.endswith(".npy") and not file.endswith("_positions.npy"):
                    self.data.append(os.path.join(path, file))
                    self.targets.append(int(file[:-4]))
                    self.position.append(os.path.join(path, file[:-4] + "_positions.npy"))

    def __getitem__(self, index):
        events = np.load(self.data[index])
        position = center_roi_positions(np.load(self.position[index]))
        if self.transform is not None:
            events = self.transform(events)
        if self.position_transform is not None:
            position = self.position_transform(position)
        return {"data": events, "pos": position}, self.targets[index]

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return True


def _build_tonic_dataset(train: bool, frame_kw: dict, root: str) -> DVSGestureROI:
    roi_tf = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.Downsample(sensor_size=(ROI_SIZE, ROI_SIZE), target_size=(ROI_SIZE, ROI_SIZE)),
        transforms.ToFrame(sensor_size=(ROI_SIZE, ROI_SIZE, 2), **frame_kw),
    ])
    # position events (128x128 sensor coords) -> [T, 1, 32, 32] position map
    pos_tf = transforms.Compose([
        transforms.Downsample(sensor_size=(128, 128, 1), target_size=(ROI_SIZE, ROI_SIZE)),
        transforms.ToFrame(sensor_size=(ROI_SIZE, ROI_SIZE, 1), **frame_kw),
    ])
    return DVSGestureROI(root, train=train, transform=roi_tf, position_transform=pos_tf)


def _to_numpy(dataset: DVSGestureROI, n_keep: Optional[int]):
    """tonic dataset -> (roi [N,T,32,32,2], pos_map [N,T,32,32,1], y [N], kept indices [N]).
    n_keep crops every recording to its first n_keep frames (delta_t framing)."""
    rois, maps, ys, kept = [], [], [], []
    n_skipped = n_short = 0
    for i in range(len(dataset)):
        try:
            x, y = dataset[i]
        except Exception as exc:  # unreadable / corrupted recording -> skip
            n_skipped += 1
            print(f"[data] skip sample {i}: {type(exc).__name__}: {exc}")
            continue
        roi = np.transpose(np.asarray(x["data"]), (0, 2, 3, 1))   # [T,2,H,W] -> [T,H,W,2]
        pos = np.transpose(np.asarray(x["pos"]), (0, 2, 3, 1))    # [T,1,H,W] -> [T,H,W,1]

        if n_keep is not None:
            # roi and position map are framed independently: both must cover the window
            if roi.shape[0] < n_keep or pos.shape[0] < n_keep:
                n_short += 1
                continue
            roi, pos = roi[:n_keep], pos[:n_keep]
        elif pos.shape[0] != roi.shape[0]:
            T = min(pos.shape[0], roi.shape[0])
            roi, pos = roi[:T], pos[:T]

        rois.append(roi.astype(np.float32))
        maps.append(pos.astype(np.float32))
        ys.append(int(np.asarray(y)))
        kept.append(i)

    if n_skipped:
        print(f"[data] {n_skipped} unreadable recordings skipped")
    if n_short:
        print(f"[data] {n_short} recordings shorter than {n_keep} frames skipped")
    return np.stack(rois), np.stack(maps), np.asarray(ys, dtype=np.int64), np.asarray(kept, dtype=np.int64)


# ------------------------------- Loading -------------------------------------

def load_split(
    split: str,
    framing: Optional[str] = None,
    frames: int = DEFAULT_FRAMES,
    delta_t: int = DEFAULT_DELTA_T,
    root: str = DEFAULT_ROOT,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> dict:
    """
    Load one split as NumPy arrays.

    Args:
        split:     "train" or "test".
        framing:   "frames" or "delta_t"; default "frames" for train, "delta_t" for test.
        frames:    number of frames per recording with "frames" framing.
        delta_t:   frame length in microseconds with "delta_t" framing.
        root:      folder containing DVSGestureROI_fs32/.
        cache_dir: the framed arrays are saved here as .npz and reused.

    Returns:
        dict with
          "roi":        [N, T, 32, 32, 2] float32  ROI event frames
          "pos_map":    [N, T, 32, 32, 1] float32  ROI position map, aligned with "roi"
          "pos_events": list of N structured arrays (x, y, s, p, t), raw position events
          "y":          [N] int64 labels
    """
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    if framing is None:
        framing = "frames" if split == "train" else "delta_t"
    if framing == "frames":
        frame_kw, n_keep, tag = {"n_time_bins": int(frames)}, None, f"fr{frames}"
    elif framing == "delta_t":
        frame_kw, n_keep, tag = {"time_window": int(delta_t)}, n_window_frames(delta_t), f"dt{delta_t}"
    else:
        raise ValueError("framing must be 'frames' or 'delta_t'")

    ds = _build_tonic_dataset(split == "train", frame_kw, root)
    cache_file = Path(cache_dir) / f"roigesture_{split}_{tag}.npz"
    if cache_file.exists():
        d = np.load(cache_file)
        roi, pos_map, y, kept = d["roi"], d["pos_map"], d["y"], d["idx"]
    else:
        print(f"[data] building roigesture/{split} ({tag}) - first time only, it takes a while")
        roi, pos_map, y, kept = _to_numpy(ds, n_keep)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, roi=roi, pos_map=pos_map, y=y, idx=kept)
        print(f"[data] saved {cache_file}  roi={roi.shape} pos_map={pos_map.shape}")

    pos_events = [center_roi_positions(np.load(ds.position[i])) for i in kept]
    return {"roi": roi, "pos_map": pos_map, "pos_events": pos_events, "y": y}


def load_dataset(**kwargs) -> Tuple[dict, dict]:
    """Load train and test split (keyword arguments as in load_split): returns (train, test)."""
    return load_split("train", **kwargs), load_split("test", **kwargs)


# ------------------------------- Helpers -------------------------------------

def train_val_split(split: dict, val_fraction: float = 0.2, seed: int = 0) -> Tuple[dict, dict]:
    """Stratified recording-level split of a split dict (never split frames of the
    same recording between train and validation)."""
    y = split["y"]
    rng = np.random.default_rng(seed)
    val_idx = []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        val_idx.extend(idx[:max(1, int(round(len(idx) * val_fraction)))])
    val_mask = np.zeros(len(y), dtype=bool)
    val_mask[val_idx] = True

    def take(mask):
        return {
            "roi": split["roi"][mask],
            "pos_map": split["pos_map"][mask],
            "pos_events": [e for e, m in zip(split["pos_events"], mask) if m],
            "y": y[mask],
        }
    return take(~val_mask), take(val_mask)
