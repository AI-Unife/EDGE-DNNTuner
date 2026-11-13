"""
DVSGesture ROI challenge - official metric (framework independent).

The network classifies one frame at a time. The prediction for a recording is the
class with the highest per-frame probability averaged over its T frames (soft
majority vote), exactly as in the Symbolic DNN-Tuner.

    frame_probs: [N, T, 11] per-frame class probabilities (softmax) for the N test
                 recordings, in the order returned by data.load_split("test")
    y:           [N] integer labels
"""
from __future__ import annotations

import numpy as np


def recording_predictions(frame_probs: np.ndarray) -> np.ndarray:
    """[N, T, C] per-frame probabilities -> [N] predicted class per recording."""
    frame_probs = np.asarray(frame_probs, dtype=np.float32)
    if frame_probs.ndim != 3:
        raise ValueError(f"expected [N, T, C] per-frame probabilities, got {frame_probs.shape}")
    return frame_probs.mean(axis=1).argmax(axis=1)


def accuracy(frame_probs: np.ndarray, y: np.ndarray) -> float:
    """Recording-level accuracy (soft majority vote over frames)."""
    return float((recording_predictions(frame_probs) == np.asarray(y)).mean())


def confusion_matrix(frame_probs: np.ndarray, y: np.ndarray, n_classes: int = 11) -> np.ndarray:
    preds = recording_predictions(frame_probs)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y), preds), 1)
    return cm
