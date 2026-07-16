"""
test_experiment.py  –  Evaluate a pre-trained experiment and optionally retrain it.

Usage:
    python test_experiment.py --experiment <path/to/experiment_dir> [options]

Steps:
  1) Load  Model/best-model.keras
  2) Load  the dataset described in the experiment's config.yaml
  3) Test  the saved model  (accuracy and loss)
  4) Find  the best iteration from algorithm_logs/hyper-neural.txt
           using score_report.txt as the index (acc_report.txt as fallback)
  5) Rebuild the same architecture with the configured backend (tf or torch)
  6) Retrain and test  (only when --retrain is passed)
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_roi_frame_size(experiment: Path, saved_model=None) -> int:
    """
    Detect the spatial frame size (typically 16 or 32) used when preprocessing
    an ROI gesture dataset.

    Detection order:
      1. From the loaded Keras model's input shape — the most reliable source:
         the model was trained with frames of exactly that size.
      2. From any SLURM .out file in the experiment or cwd: searches for the
         first occurrence of "Building model with input_shape=(H," and reads H.
      3. Default: 32.
    """
    # 1. From loaded Keras model: input_shape is (None, H, W, C) → H = frame_size
    if saved_model is not None:
        try:
            shape = saved_model.input_shape
            # Multi-input models (ROI) expose a list of shapes; take the first branch
            if isinstance(shape, list):
                shape = shape[0]
            frame_size = int(shape[1])
            print(f"[ROI frame size] Detected={frame_size} from saved model input shape {shape}.")
            return frame_size
        except Exception:
            pass

    # 2. From .out file — all iterations share the same frame_size, so the first
    #    match in the file is sufficient.
    pattern = re.compile(r"Building model with input_shape=\((\d+)")
    for d in [experiment, Path(".")]:
        for out_file in sorted(d.glob("*.out")):
            try:
                text = out_file.read_text(errors="replace")
            except OSError:
                continue
            match = pattern.search(text)
            if match:
                frame_size = int(match.group(1))
                print(f"[ROI frame size] Detected={frame_size} from '{out_file.name}'.")
                return frame_size

    print("[ROI frame size] Could not detect — using default=32.")
    return 32


def _load_dataset(dataset_name: str, frame_size: int = 32):
    """Instantiate and load the correct TunerDataset for the given dataset name."""
    from components.dataset import TunerDataset

    ds = TunerDataset()
    name = dataset_name.lower()

    if name == "cifar10":
        ds.load_cifar_10()
    elif name == "cifar100":
        ds.load_cifar_100()
    elif name == "mnist":
        ds.load_mnist()
    elif name in ("cifar10_light", "light_cifar", "light"):
        ds.load_light_cifar()
    elif name == "gesture":
        ds.load_gesture()
    elif "roigesture" in name:
        ds.load_roi_gesture(frame_size=frame_size)
    elif name == "tinyimagenet":
        ds.load_tiny_imagenet()
    elif name == "cca":
        ds.load_cca()
    elif name == "cim":
        ds.load_cim()
    elif name == "beans":
        ds.load_beans()
    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            "Supported: cifar10, cifar100, mnist, beans, light, gesture, "
            "roigesture_*, tinyimagenet, cca, cim."
        )
    return ds


def _parse_best_from_out(
    experiment: Path,
) -> "tuple[dict, int, int] | None":
    """
    Parse a SLURM .out log and extract the best iteration's hyperparameters,
    layer_x_block, and 0-based iteration index — all from a single source of truth.

    Each iteration block (delimited by ``--- ITERATION N ---`` banners) contains:
        Chosen point: {<params dict>}
        Building model with input_shape=(...), ..., layer_x_block=N
        ...
        ACCURACY: <val>
        SCORE: <val>

    Ranking:
      - SCORE  is available → minimise (lower combined score is better)
      - SCORE  not available → maximise ACCURACY

    Search order: experiment directory first, then current working directory.

    Returns:
        (params_dict, layer_x_block, iteration_index_0based)
        or None if no parsable .out file is found.
    """
    for d in [experiment, Path(".")]:
        for out_file in sorted(d.glob("*.out")):
            try:
                raw = out_file.read_text(errors="replace")
            except OSError:
                continue

            # Strip ANSI escape codes so regexes work cleanly
            text = re.sub(r"\x1b\[[0-9;]*m", "", raw)

            # Split into per-iteration blocks; blocks[0] = header
            blocks = re.split(r"---\s*ITERATION\s+\d+\s*---", text)
            if len(blocks) < 2:
                continue  # no iterations found in this file

            iterations = []
            for idx, block in enumerate(blocks[1:]):  # idx = 0-based iteration index
                # --- hyperparameters ---
                m = re.search(r"Chosen point:\s*(\{.+?\})", block, re.DOTALL)
                if not m:
                    continue
                try:
                    params = ast.literal_eval(m.group(1))
                except Exception:
                    continue

                # --- architecture ---
                lxb_m = re.search(r"layer_x_block=(\d+)", block)
                lxb = int(lxb_m.group(1)) if lxb_m else 2

                # --- scores ---
                score_m = re.search(r"SCORE:\s*([-\d.eE+]+)", block)
                acc_m   = re.search(r"ACCURACY:\s*([-\d.eE+]+)", block)
                score = float(score_m.group(1)) if score_m else None
                acc   = float(acc_m.group(1))   if acc_m   else None

                iterations.append(
                    {"index": idx, "params": params, "lxb": lxb,
                     "score": score, "acc": acc}
                )

            if not iterations:
                continue

            # Rank: prefer SCORE (lower = better), fall back to ACCURACY (higher = better)
            has_score = any(it["score"] is not None for it in iterations)
            valid = [it for it in iterations
                     if (it["score"] if has_score else it["acc"]) is not None]
            if not valid:
                continue

            if has_score:
                best = min(valid, key=lambda it: it["score"])
                print(
                    f"[Selection] '{out_file.name}' — "
                    f"best iteration (0-based): {best['index']} "
                    f"(score={best['score']:.4f})"
                )
            else:
                best = max(valid, key=lambda it: it["acc"])
                print(
                    f"[Selection] '{out_file.name}' — "
                    f"best iteration (0-based): {best['index']} "
                    f"(acc={best['acc']:.4f})"
                )

            print(f"[Hyperparams]    {best['params']}")
            print(f"[layer_x_block]  {best['lxb']}")
            return best["params"], best["lxb"], best["index"]

    return None  # no .out file found / parsable


def _find_best_iteration(algo_logs: Path) -> int:
    """Fallback: return the 0-based index of the best iteration from log files."""
    score_path = algo_logs / "score_report.txt"
    acc_path = algo_logs / "acc_report.txt"

    if score_path.exists():
        # Parse all non-empty, non-"None" lines as floats
        values = [
            float(l.strip())
            for l in score_path.read_text().splitlines()
            if l.strip() and l.strip().lower() != "none"
        ]
        if not values:
            raise ValueError(f"score_report.txt is empty in {algo_logs}")
        # The best iteration has the lowest combined score
        best_idx = min(range(len(values)), key=lambda i: values[i])
        print(
            f"[Selection] score_report.txt — best iteration: {best_idx} "
            f"(score={values[best_idx]:.4f})"
        )
    elif acc_path.exists():
        # Fallback: use validation accuracy (higher is better)
        values = [
            float(l.strip())
            for l in acc_path.read_text().splitlines()
            if l.strip() and l.strip().lower() != "none"
        ]
        if not values:
            raise ValueError(f"acc_report.txt is empty in {algo_logs}")
        best_idx = max(range(len(values)), key=lambda i: values[i])
        print(
            f"[Selection] acc_report.txt — best iteration: {best_idx} "
            f"(acc={values[best_idx]:.4f})"
        )
    else:
        raise FileNotFoundError(
            f"Neither score_report.txt nor acc_report.txt found in {algo_logs}"
        )

    return best_idx


def _load_best_params(algo_logs: Path, best_idx: int) -> dict:
    """Read line best_idx from hyper-neural.txt and parse it into a dict."""
    hyper_path = algo_logs / "hyper-neural.txt"
    if not hyper_path.exists():
        raise FileNotFoundError(f"hyper-neural.txt not found in {algo_logs}")

    # Each line is a Python dict literal written by ObjectiveWrapper.objective()
    lines = [l.strip() for l in hyper_path.read_text().splitlines() if l.strip()]
    if best_idx >= len(lines):
        raise IndexError(
            f"best_idx={best_idx} out of range: hyper-neural.txt has {len(lines)} lines."
        )
    # Safe parse: ast.literal_eval handles plain dict literals without executing code
    params = ast.literal_eval(lines[best_idx])
    print(f"[Hyperparams] {params}")
    return params


def _find_layer_x_block(experiment: Path, best_idx: int) -> int:
    """
    Extract the layer_x_block value used at iteration best_idx from the SLURM
    .out file produced during the tuning run.

    The log line has the form:
        Building model with input_shape=(...), ..., layer_x_block=N

    Search order: experiment directory first, then the current working directory.
    Falls back to 2 if no matching .out file is found.
    """
    # Look inside the experiment dir first, then the cwd (where sbatch saves .out)
    search_dirs = [experiment, Path(".")]
    for d in search_dirs:
        out_files = sorted(d.glob("*.out"))
        for out_file in out_files:
            try:
                text = out_file.read_text(errors="replace")
            except OSError:
                continue
            # Split the log by iteration banners to isolate each iteration block
            blocks = re.split(r"---\s*ITERATION\s+\d+\s*---", text)
            if best_idx + 1 < len(blocks):
                match = re.search(r"layer_x_block=(\d+)", blocks[best_idx + 1])
                if match:
                    val = int(match.group(1))
                    print(
                        f"[layer_x_block] Found={val} in '{out_file.name}' "
                        f"(iteration {best_idx})"
                    )
                    return val

    print("[layer_x_block] Not found in any .out file — using default=2")
    return 2


def _eval_keras_model(model, dataset, cfg) -> tuple[float, float]:
    """
    Evaluate a Keras model loaded from disk against the test split.

    Handles three dataset variants:
      - Standard image datasets  (single input, integer or one-hot labels)
      - ROI gesture datasets     (dual input: [image, position map])
      - Temporal gesture datasets (frame-by-frame forward pass via test_utils)

    Returns:
        (loss, accuracy) as plain Python floats.
    """
    import tensorflow as tf

    n_classes = dataset.n_classes

    # One-hot encode integer labels to match the training target format
    y_test = dataset.Y_test
    if y_test.ndim == 1:
        y_test = tf.keras.utils.to_categorical(y_test, n_classes)

    x_test = dataset.X_test.astype("float32")

    # Compile with the same loss used during training; no optimizer needed for eval
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Detect dataset variant
    is_roi = hasattr(dataset, "pos_test") and dataset.pos_test is not None
    is_gesture = (cfg.mode in ("fwdPass", "hybrid")) and ("gesture" in cfg.dataset)

    if is_gesture:
        # Temporal evaluation: iterate over time frames and vote by majority
        from test_utils import eval_model as gesture_eval
        x_input = [x_test, dataset.pos_test] if is_roi else x_test
        score = gesture_eval(model, x_input, y_test)
    elif is_roi:
        # Dual-input model: pass both image and position map
        score = model.evaluate([x_test, dataset.pos_test.astype("float32")], y_test, verbose=2)
    else:
        # Standard single-input evaluation
        score = model.evaluate(x_test, y_test, verbose=2)

    loss_val, acc_val = float(score[0]), float(score[1])
    print(f"[Saved model evaluation] loss={loss_val:.4f}  accuracy={acc_val:.4f}")
    return loss_val, acc_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and optionally retrain a Symbolic DNN Tuner experiment."
    )
    parser.add_argument(
        "--experiment", required=True,
        help="Path to the experiment directory (must contain config.yaml)."
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override the number of training epochs (optional)."
    )
    parser.add_argument(
        "--backend", type=str, default=None, choices=["tf", "torch"],
        help="Override the backend framework for retraining: 'tf' or 'torch' (optional)."
    )
    parser.add_argument(
        "--retrain", action="store_true", default=False,
        help="Rebuild the best architecture and retrain from scratch (steps 5-6). "
             "If omitted, only evaluation of the saved model is performed (steps 1-3)."
    )
    args = parser.parse_args()

    experiment = Path(args.experiment).expanduser().resolve()
    if not experiment.is_dir():
        print(f"[ERROR] Directory not found: {experiment}", file=sys.stderr)
        sys.exit(1)

    config_path = experiment / "config.yaml"
    if not config_path.exists():
        print(f"[ERROR] config.yaml not found in {experiment}", file=sys.stderr)
        sys.exit(1)

    # ── 0. Activate the experiment config ────────────────────────────────────
    # exp_config uses an environment variable (EXP_CONFIG) to locate the active
    # config.yaml; set_active_config writes that variable for the current process.
    from exp_config import set_active_config, load_cfg, reload_cfg

    set_active_config(config_path)
    cfg = load_cfg(force=True)

    # Apply CLI overrides by patching config.yaml in-place and reloading.
    # This ensures that any component reading load_cfg() picks up the new values.
    if args.epochs is not None or args.backend is not None:
        import yaml

        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)
        if args.epochs is not None:
            raw["epochs"] = args.epochs
        if args.backend is not None:
            raw["backend"] = args.backend
        with open(config_path, "w") as f:
            yaml.safe_dump(raw, f, sort_keys=False, allow_unicode=True)
        cfg = reload_cfg()

    print(f"\n{'='*60}")
    print(f"Experiment  : {experiment.name}")
    print(f"Dataset     : {cfg.dataset}")
    print(f"Backend     : {cfg.backend}")
    print(f"Epochs      : {cfg.epochs}")
    print(f"Retrain     : {args.retrain}")
    print(f"{'='*60}\n")

    # ── 1. Load the saved Keras model ────────────────────────────────────────
    # load_keras_model (test_utils) registers LayerWiseLR as a custom object so
    # that models saved with that optimizer can be deserialized correctly.
    keras_model_path = experiment / "Model" / "best-model.keras"
    if not keras_model_path.exists():
        print(
            f"[WARNING] {keras_model_path} not found. "
            "Skipping pre-trained model evaluation."
        )
        saved_model = None
    else:
        print(f"\n[1] Loading model from: {keras_model_path}")
        from test_utils import load_keras_model
        saved_model = load_keras_model(str(keras_model_path))
        print("[1] Model loaded.")

    # ── 2. Load the dataset ───────────────────────────────────────────────────
    # For ROI gesture datasets, the spatial frame size (16 or 32) must be known
    # before loading so the correct preprocessed cache is selected.
    # Detect it from the loaded model (most reliable) or from the .out log file.
    roi_frame_size = 32
    if "roigesture" in cfg.dataset.lower():
        roi_frame_size = _detect_roi_frame_size(experiment, saved_model=saved_model)

    print(f"\n[2] Loading dataset '{cfg.dataset}'" +
          (f" (frame_size={roi_frame_size})" if "roigesture" in cfg.dataset.lower() else "") +
          "...")
    dataset = _load_dataset(cfg.dataset, frame_size=roi_frame_size)
    # Ensure float32 dtype for both frameworks (avoids silent type mismatches)
    dataset.data_as_float32()
    print(
        f"[2] Dataset loaded: "
        f"{dataset.X_train.shape[0]} train / {dataset.X_test.shape[0]} test samples."
    )

    # ── 3. Evaluate the pre-trained model ────────────────────────────────────
    if saved_model is not None:
        print("\n[3] Evaluating pre-trained model (best-model.keras)...")
        saved_loss, saved_acc = _eval_keras_model(saved_model, dataset, cfg)
    else:
        print("\n[3] No pre-trained model to evaluate.")

    # ── 4. Extract the best iteration point ──────────────────────────────────
    # Primary source: SLURM .out file — single source of truth that contains
    # hyperparameters, layer_x_block, and scores all in one place.
    # Fallback: algorithm_logs/ text files (hyper-neural.txt + score/acc_report.txt)
    print(f"\n[4] Finding best iteration ...")
    out_result = _parse_best_from_out(experiment)
    if out_result is not None:
        best_params, layer_x_block, best_idx = out_result
    else:
        print("[4] No .out file found — falling back to algorithm_logs/")
        algo_logs = experiment / "algorithm_logs"
        best_idx = _find_best_iteration(algo_logs)
        best_params = _load_best_params(algo_logs, best_idx)
        layer_x_block = _find_layer_x_block(experiment, best_idx)
    print(f"[4] layer_x_block={layer_x_block}")

    # ── 5–6. Rebuild + retrain (only when --retrain is passed) ───────────────

    # cfg.name may be a *relative* path written at tuning time (e.g.
    # "results_gesture_new/26_03_...").  All file I/O inside training() builds
    # paths like "{cfg.name}/Model/..." so using a relative name breaks when
    # the script is run from a different working directory.
    # Fix: overwrite cfg.name in config.yaml with the resolved absolute path of
    # the experiment directory, then reload — training() will then always use
    # the correct absolute path regardless of the current working directory.
    if str(experiment) != cfg.name:
        import yaml
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)
        raw["name"] = str(experiment)
        with open(config_path, "w") as f:
            yaml.safe_dump(raw, f, sort_keys=False, allow_unicode=True)
        cfg = reload_cfg()
        print(f"[Config] Updated 'name' to absolute path: {experiment}")

    if not args.retrain:
        # Only print a summary of what was found and exit cleanly
        print(f"\n{'='*60}")
        print("SUMMARY  (evaluation only — pass --retrain to rebuild and retrain)")
        if saved_model is not None:
            print(f"  Pre-trained model  →  loss={saved_loss:.4f}  acc={saved_acc:.4f}")
        print(f"  Best iteration     : {best_idx}")
        print(f"  layer_x_block      : {layer_x_block}")
        print(f"  Hyperparameters    : {best_params}")
        print(f"{'='*60}\n")
        return

    # ── 5. Rebuild the architecture with the configured backend ──────────────
    print(f"\n[5] Rebuilding model with backend='{cfg.backend}'...")

    if cfg.backend == "tf":
        from tensorflow_implementation import module_backend, neural_network
        # Clear the Keras session to release GPU memory from the loaded model
        from tensorflow.keras import backend as K
        K.clear_session()
    elif cfg.backend == "torch":
        from pytorch_implementation import module_backend, neural_network
    else:
        print(f"[ERROR] Unsupported backend: {cfg.backend}", file=sys.stderr)
        sys.exit(1)

    backend_instance = module_backend.ModuleBackend()
    # NeuralNetwork wraps the framework-specific model and handles training;
    # da/reg/residual are set to False — the hyperparams dict controls them.
    nn = neural_network.NeuralNetwork(
        backend=backend_instance,
        dataset=dataset,
        da=False,
        reg=False,
        residual=False,
    )

    # build_network creates the model graph from the hyperparameter dict
    nn.build_network(best_params, layer_x_block=layer_x_block)
    print(f"[5] Model built (backend={cfg.backend}).")

    # ── 6. Retrain from scratch and evaluate ─────────────────────────────────
    print(f"\n[6] Retraining for {cfg.epochs} epoch(s)...")
    # training() compiles, fits with early stopping, and returns the best score
    score, history, trained_model = nn.training(best_params)

    retrain_loss = float(score[0])
    retrain_acc = float(score[1])
    print(f"\n[6] Retrain results: loss={retrain_loss:.4f}  accuracy={retrain_acc:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    if saved_model is not None:
        print(f"  Pre-trained model  →  loss={saved_loss:.4f}  acc={saved_acc:.4f}")
    print(f"  Retrained model    →  loss={retrain_loss:.4f}  acc={retrain_acc:.4f}")
    print(f"  Best iteration     : {best_idx}")
    print(f"  layer_x_block      : {layer_x_block}")
    print(f"  Hyperparameters    : {best_params}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
