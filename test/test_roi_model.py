#!/usr/bin/env python3
"""
Standalone script to test a saved model on roigesture_matrix
and roigesture_coords datasets.

Usage:
    python test_roi_model.py \
        --model_path best-model.keras \
        --dataset roigesture_matrix \
    --mode depth

The model is searched at <model_path>
"""

import argparse
import sys

import numpy as np
from sklearn import metrics
import tensorflow as tf
from test_utils import load_keras_model, load_roi_dataset, eval_model

# ---------------------------------------------------------------------------
# Parameters configurable in code (fallback when CLI is not used)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = "26_03_09_12_74378_roigesture_coords_fwdPass_32_2.keras"             # Model path to load 
DEFAULT_MODE     = "fwdPass"          # fwdPass | depth | hybrid
DEFAULT_DATASET  = "roigesture_coords"  # roigesture_matrix | roigesture_coords
FRAMES = 32
CHANNELS = 2

# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test a Keras model on roigesture_matrix / roigesture_coords datasets."
    )
    parser.add_argument(
        "--model_path", type=str, default=DEFAULT_MODEL_PATH,
        help="Experiment folder containing Model/best-model.keras"
    )
    parser.add_argument(
        "--dataset", type=str, default=DEFAULT_DATASET,
        choices=["roigesture_matrix", "roigesture_coords"],
        help="Dataset to use for testing"
    )
    parser.add_argument(
        "--mode", type=str, default=DEFAULT_MODE,
        choices=["fwdPass", "depth", "hybrid"],
        help="Model mode"
    )
    parser.add_argument(
        "--polarity", type=str, default="both",
        choices=["both", "sum", "sub", "drop"],
        help="Polarity for event-based dataset"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def evaluate(model: tf.keras.Model, mode: str, dataset_name: str,
             X_test: np.ndarray, pos_test, Y_test: np.ndarray):
    """
    Evaluate model on test set.

    - fwdPass / hybrid: uses custom temporal eval loop (eval_model)
    - depth: uses standard model.evaluate
    """
    is_roi = pos_test is not None

    if mode in ("fwdPass", "hybrid"):
        # Input: [B, T, H, W, C] and optionally [B, T, H, W, C_pos]
        X_input = [X_test, pos_test] if is_roi else X_test
        loss, acc = eval_model(model, X_input, Y_test)
        print(f"\n=== Test results ({dataset_name} | mode={mode}) ===")
        print(f"  Loss     : {loss:.4f}")
        print(f"  Accuracy : {acc:.4f}  ({acc * 100:.2f} %)")

    else:  # depth
        # Input: [B, H, W, C]
        if is_roi:
            X_input = [X_test, pos_test]
            loss, acc = model.evaluate(X_input, Y_test, verbose=2)
        else:
            loss, acc = model.evaluate(X_test, Y_test, verbose=2)
        print(f"\n=== Test results ({dataset_name} | mode={mode}) ===")
        print(f"  Loss     : {loss:.4f}")
        print(f"  Accuracy : {acc:.4f}  ({acc * 100:.2f} %)")

    return loss, acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Argument validation
    if not args.model_path:
        print("[ERROR] Please provide --model_path with the experiment folder.", file=sys.stderr)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 1) Load model
    # -----------------------------------------------------------------------
    try:
        model = load_keras_model(args.model_path)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    model.summary()

    # -----------------------------------------------------------------------
    # 2) Load dataset (no cfg file, shape inferred from model)
    # -----------------------------------------------------------------------
    print(f"\nLoading dataset '{args.dataset}' (mode={args.mode}) ...")
    X_train, pos_train, Y_train, X_test, pos_test, Y_test, meta = load_roi_dataset(
        frames=FRAMES,
        channels=CHANNELS,
        dataset_name=args.dataset,
        mode=args.mode,
    )

    print(
        f"  Inferred from model -> frames={meta['frames']}, channels={meta['channels']}"
    )
    print(f"  X_train : {X_train.shape}  Y_train : {Y_train.shape}")
    print(f"  X_test  : {X_test.shape}   Y_test  : {Y_test.shape}")
    if pos_test is not None:
        print(f"  pos_train: {pos_train.shape}")
        print(f"  pos_test : {pos_test.shape}")

    # -----------------------------------------------------------------------
    # 3) Evaluate
    # -----------------------------------------------------------------------
    model.compile(metrics=['accuracy'])  # Ensure model is compiled before evaluation
    evaluate(model, args.mode, args.dataset, X_test, pos_test, Y_test)


if __name__ == "__main__":
    main()
