from __future__ import annotations

from typing import Tuple, List, Callable, Optional
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tonic
import tonic.transforms as transforms
from tonic.dataset import Dataset 
from tonic.datasets import DVSGesture 
from sklearn.model_selection import train_test_split  # kept for compatibility if needed elsewhere

from exp_config import load_cfg


# ------------------------------- Targets -------------------------------------

class ToOneHotTimeCoding:
    """
    Convert an integer class label to one-hot and repeat it across time frames.

    Args:
        n_classes: total number of classes.
        n_frames: number of time frames to replicate the one-hot vector over.

    Returns:
        A tensor of shape [T, C] where T == n_frames and C == n_classes.
    """
    def __init__(self, n_classes: int, n_frames: int):
        self.n_classes = n_classes
        self.n_frames = n_frames

    def __call__(self, target: int) -> tf.Tensor:
        one_hot = tf.one_hot(target, self.n_classes)  # [C]
        # Repeat across time: list-of-T copies stacked -> [T, C]
        to_stack = [one_hot] * self.n_frames
        return tf.stack(to_stack, axis=0)


class BothPolarity:
    """
    Keep both polarity channels by concatenating them along the time axis.

    Expects [T, 2, H, W]; returns [(T*2), H, W].
    """
    def __call__(self, image: np.ndarray) -> np.ndarray:
        assert image.ndim == 4 and image.shape[1] == 2, f"Expected [T,2,H,W], got {image.shape}"
        T, C, H, W = image.shape  # C == 2
        # Reshape (T, 2, H, W) -> (T*2, H, W) without relying on cfg.FRAMES/64
        return image.transpose(0, 2, 3, 1).reshape(T * C, H, W).transpose(0, 1, 2)


# ------------------------------- Datasets ------------------------------------

class DVSGestureROI(DVSGesture):
    """A modified version of the DVSGesture eventset that returns regions-of-interest (ROIs) and their coordinates."""

    dtype = np.dtype(
        [("x", np.int16), ("y", np.int16), ("p", bool), ("t", np.int64)]
    )
    dtype_position = np.dtype(
        [
            ("x", np.int16),
            ("y", np.int16),
            ("s", np.int16),
            ("p", bool),
            ("t", np.float32),
        ]
    )
    ordering = dtype.names

    def __init__(
        self,
        save_to: str,
        output_size: Tuple,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        position_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        Dataset.__init__(
            self,
            save_to,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        
        self.output_size = output_size
        self.location_on_system = (
            self.location_on_system + f"_fs{self.output_size[0]}"
        )
        self.position_transform = position_transform
        self.train = train

        if train:
            self.folder_name = "train"
        else:
            self.folder_name = "test"

        self.users = []
        self.lighting = []
        self.position = []
        file_path = os.path.join(self.location_on_system, self.folder_name)
        for path, dirs, files in os.walk(file_path):
            rel_path = os.path.relpath(path, file_path)
            if rel_path != ".":
                user, lighting = rel_path.split("_", 1)
                user = int(user[4:])
                dirs.sort()
                for file in files:
                    if file.endswith(".npy"):
                        if file.endswith("_positions.npy"):
                            continue
                        self.data.append(os.path.join(path, file))
                        self.targets.append(int(file[:-4]))
                        pos_file = file.removesuffix(".npy") + (
                            "_positions.npy"
                        )
                        self.position.append(os.path.join(path, pos_file))
                        self.users.append(user)
                        self.lighting.append(lighting)

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target, position) where target is the index of the target class and position is the position of the ROI in the input.
        """
        events = np.load(self.data[index])
        target = self.targets[index]
        position = np.load(self.position[index])
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.position_transform is not None:
            position = self.position_transform(position)
        if self.transforms is not None:
            events, target, position = self.transforms(
                events, target, position
            )
        x = {"data": events, "pos": position}
        return x, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return (
            self._is_file_present()
            and self._folder_contains_at_least_n_files_of_type(100, ".npy")
        )


class ROIMapTransform(object):
    """Transform ROI position events into frames for downstream processing"""

    def __init__(
        self,
        full_input_size: tuple = (128, 128, 2),  # DVSGesture size
        output_size: tuple = (32, 32, 1),  # Target size for downstream
        time_window: int = None,
        n_time_bins: int = None,
    ):
        self.downsample_transform = transforms.Downsample(
            sensor_size=full_input_size[:2] + (1,), target_size=output_size[:2]
        )
        if time_window is not None:  
            self.frame_transform = transforms.ToFrame(
                sensor_size=output_size,
                time_window=time_window,
            )
        elif n_time_bins is not None:
            self.frame_transform = transforms.ToFrame(
                sensor_size=output_size,
                n_time_bins=n_time_bins,
            )
        else:
            self.frame_transform = transforms.ToFrame(
                sensor_size=output_size,
                time_window=1000,
                )
        self.transform = transforms.Compose(
            [self.downsample_transform, self.frame_transform]
        )

    def __call__(self, events):
        return self.transform(events)


# ----------------------------- Numpy adapters --------------------------------

def reshape_x_pos(arr: np.ndarray, pos: Optional[np.ndarray], cfg) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Reshape x and optionally pos arrays to match model input expectations.

    Args:
        arr: Input data array.
        pos: Optional position array (may be None for non-ROI datasets).
        cfg: Configuration object with mode and channels attributes.

    Returns:
        A tuple (x_reshaped, pos_reshaped_or_None).
    """
    if cfg.mode == "fwdPass":
        # [T, 2, H, W] -> [T, H, W, 2]
        x_out = np.transpose(arr, (0, 2, 3, 1))
        pos_out = np.transpose(pos, (0, 2, 3, 1)) if pos is not None else None
        return x_out, pos_out

    elif cfg.mode == "hybrid":
        # Expect BothPolarity OR a prior framing such that arr has shape [T, 2, H, W]
        # We reshape into groups of NUM_CHANNELS along time and stack polarities as channels.
        T, C, H, W = arr.shape  # C should be 2
        assert T % cfg.channels == 0, f"T={T} not divisible by NUM_CHANNELS={cfg.channels}"
        # Group frames: (T//NUM_CHANNELS, NUM_CHANNELS, C, H, W)
        grouped = arr.reshape(T // cfg.channels, cfg.channels, C, H, W)
        # Move to (T', H, W, NUM_CHANNELS, C)
        transposed = grouped.transpose(0, 3, 4, 1, 2)
        # Merge channel dims -> (T', H, W, NUM_CHANNELS*C) == (T', H, W, 2*NUM_CHANNELS)
        x_final = transposed.reshape(T // cfg.channels, H, W, cfg.channels * C)
        # If a pos map is provided and looks like frames, try a simple transpose to keep it aligned.
        pos_out = None
        if pos is not None and hasattr(pos, "ndim") and pos.ndim == 4:
            try:
                T, C, H, W = pos.shape  # C should be 2
                assert T % cfg.channels == 0, f"T={T} not divisible by NUM_CHANNELS={cfg.channels}"
                # Group frames: (T//NUM_CHANNELS, NUM_CHANNELS, C, H, W)
                grouped = pos.reshape(T // cfg.channels, cfg.channels, C, H, W)
                # Move to (T', H, W, NUM_CHANNELS, C)
                transposed = grouped.transpose(0, 3, 4, 1, 2)
                # Merge channel dims -> (T', H, W, NUM_CHANNELS*C) == (T', H, W, 2*NUM_CHANNELS)
                pos_out = transposed.reshape(T // cfg.channels, H, W, cfg.channels * C)
            except Exception:
                pos_out = None
        return x_final, pos_out

    else:  # "depth" or any other single-frame mode
        # [T, 2, H, W] -> assume polarity transform reduced to [T, H, W] or similar
        # If still [T, 2, H, W], collapse polarity by sum; otherwise, keep as is.
        # print(f"Reshaping for depth mode: input shape {arr.shape}")
        if arr.ndim == 4 and arr.shape[1] == 2:
            arr = arr.sum(axis=1)  # simple collapse to [H, W] (choose your policy)
        # Ensure shape is [H, W] or [H, W, C]
        if arr.ndim == 2:
            arr = arr[..., None]  # [H, W, 1]
        elif arr.ndim == 3 and arr.shape[0] not in (1, 2, 3, 4):  # likely [T, H, W]
            arr = np.transpose(arr, (1, 2, 0))
        # print(f"Reshaping for depth mode: pos shape {pos.shape}")
        if pos is not None and hasattr(pos, "ndim"):
            if pos.ndim == 4 and pos.shape[1] == 2:
                pos = pos.sum(axis=1)  # collapse polarity if still present
            if pos.ndim == 2:
                pos = pos[..., None]  # [H, W, 1]
            elif pos.ndim == 3 and pos.shape[0] not in (1, 2, 3, 4):  # likely [T, H, W]
                pos = np.transpose(pos, (1, 2, 0))
        # print(f"Reshaping for depth mode: output shape {arr.shape}, pos shape {pos.shape if pos is not None else None}")
        return arr, pos

def dataset_to_numpy(dataset, cfg, ragged: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a (cached) tonic dataset into NumPy arrays (or object array for ROI) with the layout expected by the model.

    Output X layout depends on cfg.MODE:
      - "fwdPass":    input per time step with 2 channels -> [B, T, H, W, 2]
      - "hybrid":     merge groups of NUM_CHANNELS frames and both polarities -> [B, T', H, W, 2*NUM_CHANNELS]
      - "depth":      single frame with possibly combined polarity -> [B, H, W, C]
    For ROI datasets, each element of X is a dict {"data": <array>, "pos": <array>} and the returned X
    is an object array with shape (N,). For non-ROI, X is a numeric array.

    ``ragged`` selects the fwdPass label/length convention:
      - ragged=True  (fixed-Δt framing): each recording keeps its own number of
        frames T_i and y is one integer label per recording ([B]); the [T_i, C]
        one-hot is materialised later, per sample, by the frame-by-frame evaluator.
      - ragged=False (fixed number of frames): every recording has T == cfg.frames
        and y is the time-repeated one-hot [T, C] produced by the target_transform
        (old training convention).
    For hybrid/depth ``ragged`` is ignored (labels handled as before).
    """
    x_list, y_list = [], []

    n_total = len(dataset)
    n_skipped = 0
    for i in range(n_total):
        try:
            x, y = dataset[i]
        except Exception as exc:  # unreadable / corrupted recording -> skip, don't abort
            n_skipped += 1
            print(f"[dataset_to_numpy] skip sample {i}/{n_total}: {type(exc).__name__}: {exc}")
            continue
        # Handle ROI items (dicts with 'data' and 'pos')
        if isinstance(x, dict) or (hasattr(cfg, "dataset") and "roi" in cfg.dataset.lower()):
            events = x["data"]
            pos = x.get("pos", None)
            arr = np.array(events)
            x_reshaped, pos = reshape_x_pos(arr, np.array(pos) if pos is not None else None, cfg)
            if cfg.mode == "fwdPass" and ragged and pos is not None:
                # data and ROI-map are framed by two independent ToFrame passes and
                # can differ by a frame -> keep only their common leading length.
                T = min(x_reshaped.shape[0], pos.shape[0])
                x_reshaped, pos = x_reshaped[:T], pos[:T]
            if cfg.dataset == "roigesture_coords":
                pos_mean = None
                if pos is not None:
                    # Compute center of mass for ROI position map
                    # Handling different shapes based on cfg.mode:
                    # - fwdPass: pos.shape = [T, H, W, 1] -> pos_mean.shape = [T, 2]
                    # - hybrid:  pos.shape = [T', H, W, 2*NUM_CHANNELS] -> pos_mean.shape = [T', 2]
                    # - depth:   pos.shape = [H, W, 1] or [H, W, C] -> pos_mean.shape = [2]
                    
                    if cfg.mode == "depth":
                        # Single frame case: squeeze to [H, W] and compute one center of mass
                        A = np.squeeze(pos)  # Can be [H, W] or [H, W, C]
                        # print(f"[DEBUG] Depth mode pos shape after squeeze: {A.shape}")
                        if A.ndim == 1:  # Edge case: if becomes 1D, reshape back
                            A = np.expand_dims(A, 0)
                        # If 3D (H, W, C), sum over channels to get [H, W]
                        if A.ndim == 3:
                            A = A.sum(axis=2)  # Sum across all channels
                        if A.ndim == 2:
                            H, W = A.shape
                            yy, xx = np.indices((H, W))
                            tot = A.sum()
                            if tot > 0:
                                mean_y = (A * yy).sum() / tot
                                mean_x = (A * xx).sum() / tot
                                pos_mean = np.array([mean_y, mean_x])  # (2,)
                            else:
                                pos_mean = np.array([0.0, 0.0])
                        # print(f"[DEBUG] Depth mode pos_mean: {pos_mean}")
                    else:
                        # Time-series case (fwdPass or hybrid): squeeze last dimension(s) to get [T, H, W]
                        # Remove any singleton dimensions except the batch/time dimension
                        A = pos.copy()
                        # For fwdPass: [T, H, W, 1] -> squeeze -> [T, H, W]
                        # For hybrid: [T', H, W, 2*NUM_CHANNELS] -> squeeze or flatten channels
                        while A.ndim > 3 and (A.shape[-1] == 1 or A.shape[1] == 1):
                            if A.shape[-1] == 1:
                                A = np.squeeze(A, axis=-1)
                            if A.shape[1] == 1:
                                A = np.squeeze(A, axis=1)
                        
                        # Now A should be [T, H, W] or [T, H, W, C]
                        if A.ndim == 4:
                            # If still 4D (hybrid case), flatten channel dimension for COM calculation
                            T, H, W, C = A.shape
                            A = A.reshape(T, H, W, -1).sum(axis=3)  # [T, H, W]
                        
                        if A.ndim == 3:
                            T, H, W = A.shape
                            yy, xx = np.indices((H, W))  # [H, W], [H, W]
                            
                            # Compute center of mass for each time step
                            tot = A.sum(axis=(1, 2))  # [T]
                            # Avoid division by zero
                            tot = np.where(tot > 0, tot, 1.0)
                            
                            mean_y = (A * yy).sum(axis=(1, 2)) / tot  # [T]
                            mean_x = (A * xx).sum(axis=(1, 2)) / tot  # [T]
                            
                            pos_mean = np.stack([mean_y, mean_x], axis=1)  # [T, 2]
                    
                    x_list.append({"data": x_reshaped, "pos": pos_mean})
            elif cfg.dataset == "roigesture_matrix":
                x_list.append({"data": x_reshaped, "pos":pos})
            else:
                print(f"ERROR: Unrecognized ROI dataset type '{cfg.dataset}'; expected 'roigesture_coords', 'roigesture_matrix'.")
                exit(-1)
        else:
            # Non-ROI: plain arrays
            arr = np.array(x)
            x_reshaped, _ = reshape_x_pos(arr, None, cfg)
            x_list.append(x_reshaped)

        if cfg.mode == "fwdPass" and ragged:
            # One label per recording; the [T_i, C] one-hot is materialised later,
            # per sample, once T_i is known (frame-by-frame forward pass).
            y_list.append(int(np.asarray(y)))
        else:
            # fixed frames -> time-repeated one-hot from the target_transform,
            # or hybrid/depth label as produced upstream.
            y_list.append(np.array(y))

    if n_skipped:
        print(f"[dataset_to_numpy] {n_skipped}/{n_total} samples skipped (unreadable); "
              f"{len(x_list)} kept.")

    # Return X as object array so that elements can be dicts (ROI) or arrays; y as numeric array
    return np.array(x_list, dtype=object), np.array(y_list)


# ---------------------------- Dataset loaders --------------------------------

# ---------- resolve dataset/cache paths with fallback ----------
def _resolve_paths(dataset_path: str | Path, cache_dir: str | Path) -> tuple[str, str]:
    """
    Ensure dataset_path and cache_dir exist. If not:
      - dataset_path -> './data'
      - cache_dir    -> './cache'
    Returns normalized absolute paths (str).
    """
    dp = Path(dataset_path)
    cd = Path(cache_dir)

    # dataset path fallback
    if not dp.exists():
        dp = Path("./data")
        dp.mkdir(parents=True, exist_ok=True)
        print(f"[info] dataset_path not found. Falling back to: {dp.resolve()}")

    # cache path fallback
    if not cd.exists():
        # cd = Path("./cache")
        cd.mkdir(parents=True, exist_ok=True)
        print(f"[info] cache_dir not found. Falling back to: {cd.resolve()}")

    return str(dp.resolve()), str(cd.resolve())

def _ensure_cache_dir(cache_dir: str) -> None:
    Path(cache_dir, "train").mkdir(parents=True, exist_ok=True)
    Path(cache_dir, "test").mkdir(parents=True, exist_ok=True)

_EMPTY = (np.array([], dtype=object), np.array([]))


def get_datasets_numpy(cfg, test_only: bool = False):
    """
    Load DVSGesture through Tonic, apply transforms, and return NumPy arrays.

    Args:
        test_only: if True, only the test split is built/converted; the train
                   split is returned empty (saves time when just evaluating).

    Returns:
        ((x_train, y_train), (x_test, y_test))
    """
    dataset_path = './data/'
    # TRAIN split framed per cfg.train_framing ("frames" -> cfg.frames, "delta_t" ->
    # cfg.delta_t); TEST split is always framed with cfg.delta_t (fwdPass).
    train_framing = getattr(cfg, "train_framing", "frames")
    train_ragged = (cfg.mode == "fwdPass") and (train_framing == "delta_t")
    train_tag = f"te{cfg.delta_t}" if train_ragged else f"fr{cfg.frames}"
    cache_dir = f"./cache/DVSGesture_{cfg.mode}_tr{train_tag}_te{cfg.delta_t}_{cfg.channels}/"
    # resolve with fallback
    dataset_path, cache_dir = _resolve_paths(dataset_path, cache_dir)
    _ensure_cache_dir(cache_dir)
    print("dataset_path:", dataset_path)
    print("cache_dir:", cache_dir)

    # fwdPass: TRAIN uses a fixed number of frames (old convention), TEST uses a
    # fixed time window Δt (variable length + per-recording majority vote).
    # hybrid/depth: same framing for both splits (unchanged).
    def _split_pipeline(split_ragged: bool):
        tfms: List = [
            transforms.Denoise(filter_time=10000),
            transforms.Downsample(sensor_size=tonic.datasets.DVSGesture.sensor_size, target_size=(64, 64)),
        ]
        if cfg.mode == "fwdPass":
            if split_ragged:
                tfms.append(transforms.ToFrame(sensor_size=(64, 64, 2), time_window=cfg.delta_t))
                target_transform = None
            else:
                tfms.append(transforms.ToFrame(sensor_size=(64, 64, 2), n_time_bins=cfg.frames))
                target_transform = ToOneHotTimeCoding(n_classes=11, n_frames=cfg.frames)
        elif cfg.mode == "hybrid":
            tfms.append(transforms.ToFrame(sensor_size=(64, 64, 2), time_window=10000))
            target_transform = ToOneHotTimeCoding(n_classes=11, n_frames=cfg.frames // cfg.channels)
        else:
            tfms.append(transforms.ToFrame(sensor_size=(64, 64, 2), time_window=10000))
            tfms.append(BothPolarity())
            target_transform = None
        return transforms.Compose(tfms), target_transform

    test_ragged = (cfg.mode == "fwdPass")
    test_tf, test_target_tf = _split_pipeline(split_ragged=test_ragged)
    test = tonic.datasets.DVSGesture(
        save_to=dataset_path, transform=test_tf, target_transform=test_target_tf, train=False
    )
    cached_test = tonic.DiskCachedDataset(test, cache_path=os.path.join(cache_dir, "test"))
    x_test, y_test = dataset_to_numpy(cached_test, cfg, ragged=test_ragged)

    if test_only:
        return _EMPTY, (x_test, y_test)

    train_tf, train_target_tf = _split_pipeline(split_ragged=train_ragged)
    train = tonic.datasets.DVSGesture(
        save_to=dataset_path, transform=train_tf, target_transform=train_target_tf, train=True
    )
    cached_train = tonic.DiskCachedDataset(train, cache_path=os.path.join(cache_dir, "train"))
    x_train, y_train = dataset_to_numpy(cached_train, cfg, ragged=train_ragged)

    return (x_train, y_train), (x_test, y_test)


def get_ROI_numpy(cfg, frame_size: int = 32, test_only: bool = False) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load a folder-structured ROI dataset via ROIDataset and return NumPy arrays.

    Args:
        test_only: if True, only the test split is built/converted.

    Returns:
        ((x_train, y_train), (x_test, y_test))
    """
    dataset_path = "rois_and_coordinates/datasets/"
    # TRAIN split framed per cfg.train_framing ("frames" -> cfg.frames, "delta_t" ->
    # cfg.delta_t); TEST split is always framed with cfg.delta_t (fwdPass).
    train_framing = getattr(cfg, "train_framing", "frames")
    train_ragged = (cfg.mode == "fwdPass") and (train_framing == "delta_t")
    train_tag = f"te{cfg.delta_t}" if train_ragged else f"fr{cfg.frames}"
    cache_dir = f"./cache/DVS_ROI_{frame_size}_{cfg.mode}_tr{train_tag}_te{cfg.delta_t}_{cfg.channels}/"
    output_size = (frame_size, frame_size, 2)
    _ensure_cache_dir(cache_dir)
    print("cache_dir:", cache_dir)

    # fwdPass: TRAIN framing depends on cfg.train_framing, TEST = fixed Δt
    # (variable length). Events and the ROI map use the SAME framing so their
    # per-recording frame counts line up. hybrid/depth: unchanged.
    def _split_pipeline(split_ragged: bool):
        tfms: List = [
            transforms.Denoise(filter_time=10000),
            # Downsample and ToFrame expect 2D spatial sizes (H, W) only, not including polarity
            transforms.Downsample(sensor_size=(32, 32), target_size=(frame_size, frame_size)),
        ]
        if cfg.mode == "fwdPass":
            if split_ragged:
                tfms.append(transforms.ToFrame(sensor_size=(frame_size, frame_size, 2), time_window=cfg.delta_t))
                pos_tf = ROIMapTransform(time_window=cfg.delta_t, output_size=(frame_size, frame_size, 1))
                target_transform = None
            else:
                tfms.append(transforms.ToFrame(sensor_size=(frame_size, frame_size, 2), n_time_bins=cfg.frames))
                pos_tf = ROIMapTransform(n_time_bins=cfg.frames, output_size=(frame_size, frame_size, 1))
                target_transform = ToOneHotTimeCoding(n_classes=11, n_frames=cfg.frames)
        elif cfg.mode == "hybrid":
            tfms.append(transforms.ToFrame(sensor_size=(frame_size, frame_size, 2), time_window=10000))
            pos_tf = ROIMapTransform(time_window=10000, output_size=(frame_size, frame_size, 1))
            target_transform = ToOneHotTimeCoding(n_classes=11, n_frames=cfg.frames // cfg.channels)
        else:
            tfms.append(transforms.ToFrame(sensor_size=(frame_size, frame_size, 2), time_window=10000))
            tfms.append(BothPolarity())
            pos_tf = ROIMapTransform(time_window=10000, output_size=(frame_size, frame_size, 1))
            target_transform = None
        return transforms.Compose(tfms), target_transform, pos_tf

    test_ragged = (cfg.mode == "fwdPass")
    test_tf, test_target_tf, test_pos_tf = _split_pipeline(split_ragged=test_ragged)
    test = DVSGestureROI(
        dataset_path,
        output_size=output_size,
        train=False,
        transform=test_tf,
        target_transform=test_target_tf,
        position_transform=test_pos_tf,
    )
    cached_test = tonic.DiskCachedDataset(test, cache_path=os.path.join(cache_dir, "test"))
    x_test, y_test = dataset_to_numpy(cached_test, cfg, ragged=test_ragged)

    if test_only:
        return _EMPTY, (x_test, y_test)

    train_tf, train_target_tf, train_pos_tf = _split_pipeline(split_ragged=train_ragged)
    train = DVSGestureROI(
        dataset_path,
        output_size=output_size,
        train=True,
        transform=train_tf,
        target_transform=train_target_tf,
        position_transform=train_pos_tf,
    )
    print("Loaded ROI training dataset with", len(train), "samples.")
    cached_train = tonic.DiskCachedDataset(train, cache_path=os.path.join(cache_dir, "train"))
    x_train, y_train = dataset_to_numpy(cached_train, cfg, ragged=train_ragged)

    return (x_train, y_train), (x_test, y_test)


# ------------------------------ Public API -----------------------------------

def ROI_data():
    """Compatibility shim if external code expects this name."""
    return gesture_data(ROI=True)


def gesture_data(num_classes: int = 11, ROI: bool = False, frame_size: int = 32, test_only: bool = False):
    """
    End-to-end loader producing NumPy arrays ready for model consumption.

    ``test_only=True`` skips building the train split (returns it empty).

    Shapes:
      - fwdPass:
          X: [B, T, H, W, 2]
          y: [B, T, C]
      - hybrid:
          X: [B, T', H, W, 2*NUM_CHANNELS] with T' = T/NUM_CHANNELS
          y: [B, T', C]
      - depth:
          X: [B, H, W, C]
          y: integer labels (will be one-hot here)

    Returns:
        x_train, x_test, y_train, y_test
    """
    cfg = load_cfg()
    if ROI:
        (x_train, y_train), (x_test, y_test) = get_ROI_numpy(cfg=cfg, frame_size=frame_size, test_only=test_only)
    else:
        (x_train, y_train), (x_test, y_test) = get_datasets_numpy(cfg=cfg, test_only=test_only)

    # Convert labels to one-hot for single-frame modes
    if cfg.mode == "depth":
        if len(y_train):
            y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)


    return x_train, y_train, x_test, y_test