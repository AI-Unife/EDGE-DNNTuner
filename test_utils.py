from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Optimizer
import tonic
import tonic.transforms as transforms
from tonic.dataset import Dataset
from tonic.datasets import DVSGesture


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

@keras.utils.register_keras_serializable()
class LayerWiseLR(Optimizer):
    """
    Compatibility optimizer wrapper used only to deserialize models
    that were saved with this custom optimizer.
    """

    def __init__(self, optimizer, multiplier, learning_rate=0.001, name="LWLR", **kwargs):
        if hasattr(Optimizer, "_HAS_AGGREGATE_GRAD"):
            super().__init__(name=name, **kwargs)
            self._set_hyper("learning_rate", learning_rate)
        else:
            super().__init__(name=name, **kwargs)

        self._learning_rate = learning_rate
        self._optimizer = optimizer
        self._multiplier = multiplier

    def apply_gradients(self, grads_and_vars, name: str = None, experimental_aggregate_gradients: bool = True):
        updated = []
        for grad, var in grads_and_vars:
            if grad is not None:
                layer_name = (getattr(var, "path", None) or var.name).split("/")[0]
                scale = self._multiplier.get(layer_name, 1.0)
                updated.append((grad * scale, var))
            else:
                updated.append((grad, var))
        self._optimizer.learning_rate.assign(self._learning_rate)
        return self._optimizer.apply_gradients(updated)

    def _create_slots(self, var_list):
        if hasattr(self._optimizer, "_create_slots"):
            self._optimizer._create_slots(var_list)

    def get_config(self):
        cfg = super().get_config()
        base_opt_cfg = keras.optimizers.serialize(self._optimizer)
        mult_cfg = {
            str(k): float(v) if hasattr(v, "__float__") else v
            for k, v in self._multiplier.items()
        }
        cfg.update({
            "name": getattr(self, "name", self.__class__.__name__),
            "optimizer": base_opt_cfg,
            "multiplier": mult_cfg,
            "learning_rate": self._learning_rate,
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        name = config.pop("name", "LWLR")
        opt_cfg = config.pop("optimizer", None)
        optimizer = keras.optimizers.deserialize(opt_cfg) if isinstance(opt_cfg, dict) else opt_cfg
        mult = config.pop("multiplier", None)
        lr = config.get("learning_rate", None)
        return cls(optimizer=optimizer, multiplier=mult, learning_rate=lr, name=name, **config)


def load_keras_model(model_path: str) -> tf.keras.Model:
    """Load model from <model_path>"""
    keras_path = Path(model_path)
    if not keras_path.exists():
        raise FileNotFoundError(f"Model not found at: {keras_path}")

    model = tf.keras.models.load_model(
        str(keras_path),
        custom_objects={"LayerWiseLR": LayerWiseLR},
        compile=False,
    )
    
    
    return model


# ---------------------------------------------------------------------------
# Temporal evaluation helpers
# ---------------------------------------------------------------------------

def accuracy(outputs: tf.Tensor, targets: tf.Tensor) -> float:
    """Sequence-level accuracy by majority vote over time."""
    pred_frames = tf.argmax(outputs, axis=2, output_type=tf.int32)
    targ_frames = tf.argmax(targets, axis=2, output_type=tf.int32)

    num_classes = tf.shape(outputs)[-1]

    def row_mode(row: tf.Tensor) -> tf.Tensor:
        counts = tf.math.bincount(row, minlength=num_classes, maxlength=num_classes)
        return tf.argmax(counts, axis=0, output_type=tf.int32)

    pred_mode = tf.map_fn(row_mode, pred_frames, fn_output_signature=tf.int32)
    targ_mode = tf.map_fn(row_mode, targ_frames, fn_output_signature=tf.int32)

    acc = tf.reduce_mean(tf.cast(tf.equal(pred_mode, targ_mode), tf.float32))
    return float(acc.numpy())


def eval_model(model: tf.keras.Model, x_input, y) -> Tuple[float, float]:
    """
    Frame-by-frame evaluation of a per-frame classifier, one recording at a time.

    Each recording i has its own number of frames T_i (fixed-Δt framing), so X is
    iterated sample-wise instead of over a shared time axis:
      - the model is run on the [T_i, H, W, C] stack of frames -> [T_i, C] logits
      - the sequence prediction is the majority vote over per-frame argmax
      - the sequence loss is CategoricalCrossentropy with the (constant) clip label
        broadcast over its T_i frames

    Args:
        model:   per-frame Keras classifier (logits output).
        x_input: object/dense array of [T_i, H, W, C] frames, or [x_data, x_pos]
                 for ROI (x_pos elements are [T_i, ...] aligned to x_data).
        y:       clip labels as [B] integers, [B, C] one-hot, or [B, T, C]
                 time-repeated one-hot (label is taken as constant over time).

    Returns:
        (mean_loss, sequence_accuracy) as plain floats.
    """
    if isinstance(x_input, list) and len(x_input) == 2:
        x_data, x_pos = x_input
        is_roi = True
    else:
        x_data, x_pos, is_roi = x_input, None, False

    y = np.asarray(y)
    if y.ndim == 3:        # [B, T, C] time-repeated one-hot -> per-clip label
        y_true = y[:, 0, :].argmax(axis=1)
        n_classes = int(y.shape[2])
    elif y.ndim == 2:      # [B, C] one-hot
        y_true = y.argmax(axis=1)
        n_classes = int(y.shape[1])
    else:                  # [B] integer labels
        y_true = y.astype(int)
        n_classes = int(model.output_shape[-1])

    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    preds = np.empty(len(x_data), dtype=int)
    losses = np.empty(len(x_data), dtype=float)

    for i in range(len(x_data)):
        xi = tf.convert_to_tensor(np.asarray(x_data[i], dtype=np.float32))   # [T_i, H, W, C]
        if is_roi:
            pi = tf.convert_to_tensor(np.asarray(x_pos[i], dtype=np.float32))
            out_i = model([xi, pi], training=False)                         # [T_i, C]
        else:
            out_i = model(xi, training=False)                               # [T_i, C]
        out_i = tf.convert_to_tensor(out_i)

        frame_pred = tf.argmax(out_i, axis=-1, output_type=tf.int32).numpy()
        preds[i] = np.bincount(frame_pred, minlength=n_classes).argmax()

        t_i = int(out_i.shape[0])
        yi = tf.one_hot(np.full(t_i, y_true[i], dtype=np.int32), n_classes)
        losses[i] = float(loss_fn(yi, out_i).numpy())

    val_acc = float((preds == y_true).mean())
    val_loss = float(losses.mean())
    return val_loss, val_acc


# ---------------------------------------------------------------------------
# ROI dataset pipeline
# ---------------------------------------------------------------------------

class ToOneHotTimeCoding:
    """Converts integer target into one-hot repeated across time frames."""

    def __init__(self, n_classes: int, n_frames: int):
        self.n_classes = n_classes
        self.n_frames = n_frames

    def __call__(self, target: int) -> tf.Tensor:
        one_hot = tf.one_hot(target, self.n_classes)
        return tf.stack([one_hot] * self.n_frames, axis=0)


class BothPolarity:
    """Concatenates both polarities along the temporal axis."""

    def __call__(self, image: np.ndarray) -> np.ndarray:
        assert image.ndim == 4 and image.shape[1] == 2, f"Expected [T,2,H,W], got {image.shape}"
        t, c, _, _ = image.shape
        return image.transpose(0, 2, 3, 1).reshape(t * c, image.shape[2], image.shape[3]).transpose(0, 1, 2)


class DVSGestureROI(DVSGesture):
    """ROI version of DVSGesture returning events, target, and position map."""

    sensor_size = (32, 32, 2)
    dtype = np.dtype([("x", np.int16), ("y", np.int16), ("p", bool), ("t", np.int64)])
    dtype_position = np.dtype(
        [("x", np.int16), ("y", np.int16), ("s", np.int16), ("p", bool), ("t", np.float32)]
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
        self.location_on_system = self.location_on_system + f"_fs{self.output_size[0]}"
        self.position_transform = position_transform
        self.train = train
        self.folder_name = "train" if train else "test"

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
                        pos_file = file.removesuffix(".npy") + "_positions.npy"
                        self.position.append(os.path.join(path, pos_file))
                        self.users.append(user)
                        self.lighting.append(lighting)

    def __getitem__(self, index):
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
            events, target, position = self.transforms(events, target, position)
        x = {"data": events, "pos": position}
        return x, target

    def __len__(self):
        return len(self.data)


class ROIMapTransform:
    """Transforms ROI position events into frames."""

    def __init__(
        self,
        full_input_size: tuple = (128, 128, 2),
        output_size: tuple = (32, 32, 1),
        time_window: int = None,
        n_time_bins: int = None,
    ):
        self.downsample_transform = transforms.Downsample(
            sensor_size=full_input_size[:2] + (1,), target_size=output_size[:2]
        )
        if time_window is not None:
            self.frame_transform = transforms.ToFrame(sensor_size=output_size, time_window=time_window)
        elif n_time_bins is not None:
            self.frame_transform = transforms.ToFrame(sensor_size=output_size, n_time_bins=n_time_bins)
        else:
            self.frame_transform = transforms.ToFrame(sensor_size=output_size, time_window=1000)
        self.transform = transforms.Compose([self.downsample_transform, self.frame_transform])

    def __call__(self, events):
        return self.transform(events)


@dataclass
class LocalCfg:
    dataset: str
    mode: str
    frames: int
    channels: int


def reshape_x_pos(arr: np.ndarray, pos: Optional[np.ndarray], cfg: LocalCfg) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if cfg.mode == "fwdPass":
        x_out = np.transpose(arr, (0, 2, 3, 1))
        pos_out = np.transpose(pos, (0, 2, 3, 1)) if pos is not None else None
        return x_out, pos_out

    if cfg.mode == "hybrid":
        t, c, h, w = arr.shape
        assert t % cfg.channels == 0, f"T={t} not divisible by channels={cfg.channels}"
        grouped = arr.reshape(t // cfg.channels, cfg.channels, c, h, w)
        transposed = grouped.transpose(0, 3, 4, 1, 2)
        x_final = transposed.reshape(t // cfg.channels, h, w, cfg.channels * c)

        pos_out = None
        if pos is not None and hasattr(pos, "ndim") and pos.ndim == 4:
            try:
                t, c, h, w = pos.shape
                assert t % cfg.channels == 0, f"T={t} not divisible by channels={cfg.channels}"
                grouped = pos.reshape(t // cfg.channels, cfg.channels, c, h, w)
                transposed = grouped.transpose(0, 3, 4, 1, 2)
                pos_out = transposed.reshape(t // cfg.channels, h, w, cfg.channels * c)
            except Exception:
                pos_out = None
        return x_final, pos_out

    if arr.ndim == 4 and arr.shape[1] == 2:
        arr = arr.sum(axis=1)
    if arr.ndim == 2:
        arr = arr[..., None]
    elif arr.ndim == 3 and arr.shape[0] not in (1, 2, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if pos is not None and hasattr(pos, "ndim"):
        if pos.ndim == 4 and pos.shape[1] == 1:
            pos = pos.sum(axis=1)
        if pos.ndim == 2:
            pos = pos[..., None]
        elif pos.ndim == 3 and pos.shape[0] not in (1, 2, 3, 4):
            pos = np.transpose(pos, (1, 2, 0))

    return arr, pos


def _center_of_mass_map(pos: np.ndarray, mode: str) -> np.ndarray:
    if mode == "depth":
        a = np.squeeze(pos)
        if a.ndim == 1:
            a = np.expand_dims(a, 0)
        if a.ndim == 3:
            a = a.sum(axis=2)
        if a.ndim == 2:
            h, w = a.shape
            yy, xx = np.indices((h, w))
            tot = a.sum()
            if tot > 0:
                return np.array([(a * yy).sum() / tot, (a * xx).sum() / tot])
            return np.array([0.0, 0.0])
        return np.array([0.0, 0.0])

    a = pos.copy()
    while a.ndim > 3 and (a.shape[-1] == 1 or a.shape[1] == 1):
        if a.shape[-1] == 1:
            a = np.squeeze(a, axis=-1)
        if a.shape[1] == 1:
            a = np.squeeze(a, axis=1)

    if a.ndim == 4:
        t, h, w, _ = a.shape
        a = a.reshape(t, h, w, -1).sum(axis=3)

    if a.ndim == 3:
        t, h, w = a.shape
        yy, xx = np.indices((h, w))
        tot = a.sum(axis=(1, 2))
        tot = np.where(tot > 0, tot, 1.0)
        mean_y = (a * yy).sum(axis=(1, 2)) / tot
        mean_x = (a * xx).sum(axis=(1, 2)) / tot
        return np.stack([mean_y, mean_x], axis=1)

    return np.zeros((1, 2), dtype=np.float32)


def dataset_to_numpy(dataset, cfg: LocalCfg) -> Tuple[np.ndarray, np.ndarray]:
    x_list, y_list = [], []

    for x, y in dataset:
        events = x["data"]
        pos = x.get("pos", None)

        arr = np.array(events)
        x_reshaped, pos = reshape_x_pos(arr, np.array(pos) if pos is not None else None, cfg)

        if cfg.dataset == "roigesture_coords":
            pos_mean = _center_of_mass_map(pos, cfg.mode) if pos is not None else None
            x_list.append({"data": x_reshaped, "pos": pos_mean})
        elif cfg.dataset == "roigesture_matrix":
            x_list.append({"data": x_reshaped, "pos": pos})
        else:
            raise ValueError(f"Unsupported ROI dataset: {cfg.dataset}")

        y_list.append(np.array(y))

    return np.array(x_list, dtype=object), np.array(y_list)


def _resolve_paths(dataset_path: str | Path, cache_dir: str | Path) -> tuple[str, str]:
    dp = Path(dataset_path)
    cd = Path(cache_dir)

    if not dp.exists():
        raise FileNotFoundError(f"dataset_path not found: {dp.resolve()}")

    cd.mkdir(parents=True, exist_ok=True)
    return str(dp.resolve()), str(cd.resolve())


def _ensure_cache_dir(cache_dir: str) -> None:
    Path(cache_dir, "train").mkdir(parents=True, exist_ok=True)
    Path(cache_dir, "test").mkdir(parents=True, exist_ok=True)


def _first_input_shape(model: tf.keras.Model):
    shape = model.input_shape
    if isinstance(shape, list):
        return shape[0]
    return shape



def load_roi_dataset(
    dataset_name: str,
    mode: str,
    frames: int,
    channels: int,
    dataset_path: str = "rois_and_coordinates/datasets/",
    frame_size: int = 32,
    n_classes: int = 11,
):
    if dataset_name not in ("roigesture_matrix", "roigesture_coords"):
        raise ValueError("dataset_name must be 'roigesture_matrix' or 'roigesture_coords'")
    if mode not in ("fwdPass", "depth", "hybrid"):
        raise ValueError("mode must be fwdPass, depth, or hybrid")

    cfg = LocalCfg(dataset=dataset_name, mode=mode, frames=frames, channels=channels)

    print(f"[dataset] mode={mode}, inferred frames={frames}, inferred channels={channels}")

    cache_dir = f"./cache/DVS_ROI_fs{frame_size}_{mode}_{frames}_{channels}/"
    dataset_path, cache_dir = _resolve_paths(dataset_path, cache_dir)
    _ensure_cache_dir(cache_dir)

    output_size = (frame_size, frame_size, 2)

    tfms: List = [
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(sensor_size=output_size, n_time_bins=frames),
    ]

    if mode == "fwdPass":
        target_transform = ToOneHotTimeCoding(n_classes=n_classes, n_frames=frames)
    elif mode == "hybrid":
        target_transform = ToOneHotTimeCoding(n_classes=n_classes, n_frames=frames // channels)
    else:
        tfms.append(BothPolarity())
        target_transform = None

    transform = transforms.Compose(tfms)

    train = DVSGestureROI(
        dataset_path,
        output_size=output_size,
        train=True,
        transform=transform,
        target_transform=target_transform,
        position_transform=ROIMapTransform(n_time_bins=frames),
    )
    test = DVSGestureROI(
        dataset_path,
        output_size=output_size,
        train=False,
        transform=transform,
        target_transform=target_transform,
        position_transform=ROIMapTransform(n_time_bins=frames),
    )

    cached_train = tonic.DiskCachedDataset(train, cache_path=os.path.join(cache_dir, "train"))
    cached_test = tonic.DiskCachedDataset(test, cache_path=os.path.join(cache_dir, "test"))

    x_train_raw, y_train = dataset_to_numpy(cached_train, cfg)
    x_test_raw, y_test = dataset_to_numpy(cached_test, cfg)

    x_train = np.array([item["data"] for item in x_train_raw]).astype("float32")
    x_test = np.array([item["data"] for item in x_test_raw]).astype("float32")
    pos_train = np.array([item["pos"] for item in x_train_raw])
    pos_test = np.array([item["pos"] for item in x_test_raw])

    if mode == "depth":
        y_train = tf.keras.utils.to_categorical(y_train, n_classes)
        y_test = tf.keras.utils.to_categorical(y_test, n_classes)

    meta = {"mode": mode, "frames": frames, "channels": channels, "dataset": dataset_name}
    return x_train, pos_train, y_train, x_test, pos_test, y_test, meta
