"""
test_experiment.py  –  Evaluate a pre-trained experiment and optionally retrain it.

Usage:
    # Single experiment (must contain config.yaml directly)
    python test_experiment.py --experiment <path/to/experiment_dir> [--selection-mode MODE]

    # Batch mode (REPORT ONLY): <path> is a parent folder containing one or
    # more experiment subfolders. Any subfolder (at any depth) that contains
    # at least one SLURM ".out" file is treated as an experiment directory.
    # In this mode the script NEVER loads a dataset, NEVER loads/evaluates the
    # saved Keras model, and NEVER retrains anything — for each experiment it
    # only looks up and prints the best iteration overall and the best
    # iteration among those natively trained with activation='relu'.
    # --selection-mode/--epochs/--backend are ignored in this mode.
    python test_experiment.py --experiment <path/to/parent_dir>

Selection / activation policy (--selection-mode), chosen once per run (asked
interactively if not passed on the command line):

  native_relu         Find the best iteration AMONG THOSE that already used
                       activation='relu' natively (i.e. it was the value
                       chosen by the hyper-parameter search, not forced), and
                       retrain that architecture from scratch.

  force_relu_retrain   Find the best iteration overall (regardless of its
                       original activation), force its activation to 'relu',
                       and retrain that architecture from scratch.

  force_relu_infer     Find the best iteration overall (regardless of its
                       original activation), load the corresponding saved
                       Keras model (Model/best-model.keras), force its
                       layers' activation to 'relu' at inference time, and
                       only evaluate it (no retraining).

Steps (applied to each experiment directory):
  1) Load  the dataset described in the experiment's config.yaml
  2) Find  the best iteration (subject to the chosen selection-mode policy)
           from the SLURM .out log, or — as a fallback — from
           algorithm_logs/hyper-neural.txt using score_report.txt as the
           index (acc_report.txt as fallback)
  3) Load  Model/best-model.keras                (modes: force_relu_retrain, force_relu_infer)
  4) Test  the saved model  (accuracy and loss)   (modes: force_relu_retrain, force_relu_infer)
  5) Rebuild the same architecture with the configured backend (tf or torch)  (modes: native_relu, force_relu_retrain)
  6) Retrain and test                                                         (modes: native_relu, force_relu_retrain)
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from pathlib import Path


SELECTION_MODES = ("native_relu", "force_relu_retrain", "force_relu_infer")

SELECTION_MODE_PROMPTS = {
    "1": ("native_relu",
          "Trova il miglior modello che ha GIA' 'relu' come attivazione fin "
          "dall'origine, e riaddestra quello."),
    "2": ("force_relu_retrain",
          "Prendi il modello migliore in assoluto, imposta l'attivazione a "
          "'relu' e riaddestralo da zero."),
    "3": ("force_relu_infer",
          "Prendi il modello migliore in assoluto, imposta l'attivazione a "
          "'relu' e fai SOLO inferenza (nessun riaddestramento)."),
}


def _prompt_selection_mode() -> str:
    """Ask the user, once per run, which selection/activation policy to use."""
    print("\nSeleziona la modalita' di scelta del modello / attivazione:")
    for key, (_, desc) in SELECTION_MODE_PROMPTS.items():
        print(f"  {key}) {desc}")
    while True:
        choice = input("Scelta [1/2/3]: ").strip()
        if choice in SELECTION_MODE_PROMPTS:
            mode = SELECTION_MODE_PROMPTS[choice][0]
            print(f"[Selection mode] '{mode}' selezionato.\n")
            return mode
        print("Scelta non valida, riprova.")


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
    if saved_model is not None:
        try:
            shape = saved_model.input_shape
            if isinstance(shape, list):
                shape = shape[0]
            frame_size = int(shape[1])
            print(f"[ROI frame size] Detected={frame_size} from saved model input shape {shape}.")
            return frame_size
        except Exception:
            pass

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
    require_relu: bool = False,
) -> "tuple[dict, int, int, float, float] | None":
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

    If ``require_relu`` is True, only iterations whose original hyperparameters
    already used activation='relu' are considered candidates (this implements
    the 'native_relu' selection mode).

    Search order: experiment directory first, then current working directory.

    Returns:
        (params_dict, layer_x_block, iteration_index_0based, best_acc, best_score)
        or None if no parsable / matching .out file is found.
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
            valid = [
                it for it in iterations
                if (not require_relu or it["params"].get("activation") == "relu")
                and (it["score"] if has_score else it["acc"]) is not None
            ]
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
            return best["params"], best["lxb"], best["index"], best["acc"], best["score"]

    return None  # no .out file found / parsable / matching


def _find_best_iteration_fallback(
    algo_logs: Path, require_relu: bool = False
) -> "tuple[int, dict, str, float]":
    """
    Fallback selection (used when no .out file is available): rank iterations
    using score_report.txt (preferred, lower = better) or acc_report.txt
    (higher = better), reading the corresponding hyperparameters from
    hyper-neural.txt (same line index).

    If ``require_relu`` is True, only iterations whose hyperparameters already
    used activation='relu' are considered (the 'native_relu' selection mode).

    Returns:
        (best_idx, params_dict, metric_name, metric_value)
        where metric_name is "score" or "acc".
    """
    hyper_path = algo_logs / "hyper-neural.txt"
    if not hyper_path.exists():
        raise FileNotFoundError(f"hyper-neural.txt not found in {algo_logs}")

    lines = [l.strip() for l in hyper_path.read_text().splitlines() if l.strip()]
    all_params: "list[dict | None]" = []
    for l in lines:
        try:
            all_params.append(ast.literal_eval(l))
        except Exception:
            all_params.append(None)

    score_path = algo_logs / "score_report.txt"
    acc_path = algo_logs / "acc_report.txt"

    if score_path.exists():
        raw_lines = score_path.read_text().splitlines()
        use_score = True
        source_name = "score_report.txt"
    elif acc_path.exists():
        raw_lines = acc_path.read_text().splitlines()
        use_score = False
        source_name = "acc_report.txt"
    else:
        raise FileNotFoundError(
            f"Neither score_report.txt nor acc_report.txt found in {algo_logs}"
        )

    values: "list[float | None]" = []
    for l in raw_lines:
        l = l.strip()
        if not l or l.lower() == "none":
            values.append(None)
        else:
            values.append(float(l))

    candidates = []
    for i, v in enumerate(values):
        if v is None:
            continue
        if i >= len(all_params) or all_params[i] is None:
            continue
        if require_relu and all_params[i].get("activation") != "relu":
            continue
        candidates.append(i)

    if not candidates:
        scope = " con activation='relu'" if require_relu else ""
        raise ValueError(f"Nessuna iterazione valida{scope} trovata in {algo_logs}")

    if use_score:
        best_idx = min(candidates, key=lambda i: values[i])
        metric_name = "score"
        print(f"[Selection] {source_name} — best iteration: {best_idx} (score={values[best_idx]:.4f})")
    else:
        best_idx = max(candidates, key=lambda i: values[i])
        metric_name = "acc"
        print(f"[Selection] {source_name} — best iteration: {best_idx} (acc={values[best_idx]:.4f})")

    params = dict(all_params[best_idx])
    print(f"[Hyperparams] {params}")
    return best_idx, params, metric_name, values[best_idx]


def _find_best_overall_and_relu(experiment: Path) -> dict:
    """
    For a single experiment directory, find BOTH:
      - the best iteration overall (any activation)
      - the best iteration among those natively trained with activation='relu'

    without touching the dataset or any saved Keras model — this is used by
    batch mode, which only reports these two results and never loads data,
    loads a saved model, evaluates it, or retrains anything.

    Returns a dict:
        {
          "overall": {"idx", "params", "layer_x_block", "metric_name", "metric_value"} | None,
          "overall_error": str            # present only if "overall" is None
          "relu":    {"idx", "params", "layer_x_block", "metric_name", "metric_value"} | None,
          "relu_error": str                # present only if "relu" is None
        }
    """
    result: dict = {"overall": None, "relu": None}

    out_overall = _parse_best_from_out(experiment, require_relu=False)
    out_relu = _parse_best_from_out(experiment, require_relu=True)

    def _pack_from_out(out_result):
        params, lxb, idx, acc, score = out_result
        metric_name, metric_value = ("score", score) if score is not None else ("acc", acc)
        return {"idx": idx, "params": params, "layer_x_block": lxb,
                "metric_name": metric_name, "metric_value": metric_value}

    if out_overall is not None:
        result["overall"] = _pack_from_out(out_overall)
    if out_relu is not None:
        result["relu"] = _pack_from_out(out_relu)

    if out_overall is None or out_relu is None:
        algo_logs = experiment / "algorithm_logs"
        if out_overall is None:
            try:
                idx, params, metric_name, metric_value = _find_best_iteration_fallback(
                    algo_logs, require_relu=False
                )
                lxb = _find_layer_x_block(experiment, idx)
                result["overall"] = {"idx": idx, "params": params, "layer_x_block": lxb,
                                      "metric_name": metric_name, "metric_value": metric_value}
            except Exception as exc:
                result["overall_error"] = str(exc)
        if out_relu is None:
            try:
                idx, params, metric_name, metric_value = _find_best_iteration_fallback(
                    algo_logs, require_relu=True
                )
                lxb = _find_layer_x_block(experiment, idx)
                result["relu"] = {"idx": idx, "params": params, "layer_x_block": lxb,
                                   "metric_name": metric_name, "metric_value": metric_value}
            except Exception as exc:
                result["relu_error"] = str(exc)

    return result


def _print_candidate(label: str, cand: "dict | None", error: "str | None" = None) -> None:
    """Pretty-print a single best-iteration candidate (or the reason it's missing)."""
    if cand is None:
        msg = error or "nessuna iterazione trovata"
        print(f"  {label:<28}: NON TROVATO — {msg}")
        return
    print(
        f"  {label:<28}: iterazione={cand['idx']}  layer_x_block={cand['layer_x_block']}  "
        f"{cand['metric_name']}={cand['metric_value']:.4f}"
    )
    print(f"    hyperparams: {cand['params']}")


def _print_batch_best_group(results: "list[dict]", key: str, group_label: str) -> None:
    """
    Across all experiments in a batch, print which one obtained the best value
    for ``key`` ("overall" or "relu"), separately for score-ranked and
    accuracy-ranked experiments (the two metrics aren't comparable directly).
    """
    cands = [(r["experiment"], r[key]) for r in results if r.get(key) is not None]
    score_cands = [(e, c) for e, c in cands if c["metric_name"] == "score"]
    acc_cands = [(e, c) for e, c in cands if c["metric_name"] == "acc"]

    if not score_cands and not acc_cands:
        print(f"  {group_label}: nessun esperimento ha prodotto un risultato utilizzabile.")
        return

    if score_cands:
        best_e, best_c = min(score_cands, key=lambda x: x[1]["metric_value"])
        print(f"  {group_label} (score piu' basso)   : {best_e}  score={best_c['metric_value']:.4f}")
    if acc_cands:
        best_e, best_c = max(acc_cands, key=lambda x: x[1]["metric_value"])
        print(f"  {group_label} (accuracy piu' alta) : {best_e}  acc={best_c['metric_value']:.4f}")


def _find_layer_x_block(experiment: Path, best_idx: int) -> int:
    """
    Extract the layer_x_block value used at iteration best_idx from the SLURM
    .out file produced during the tuning run.

    Search order: experiment directory first, then the current working directory.
    Falls back to 2 if no matching .out file is found.
    """
    search_dirs = [experiment, Path(".")]
    for d in search_dirs:
        out_files = sorted(d.glob("*.out"))
        for out_file in out_files:
            try:
                text = out_file.read_text(errors="replace")
            except OSError:
                continue
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


def _force_relu_activations(model) -> None:
    """
    Monkey-patch every layer's stored activation function to ReLU, in place.

    This is used only by the 'force_relu_infer' selection mode: it swaps the
    activation of an ALREADY TRAINED model at inference time (weights are not
    retrained/recompiled), purely to measure how the saved checkpoint behaves
    if its activation had been 'relu'.
    """
    import tensorflow as tf

    changed = 0
    for layer in model.layers:
        if hasattr(layer, "activation") and layer.activation is not None:
            layer.activation = tf.keras.activations.get("relu")
            changed += 1
    print(f"[Force ReLU] Patched activation on {changed} layer(s) of the saved model.")


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

    y_test = dataset.Y_test
    if y_test.ndim == 1:
        y_test = tf.keras.utils.to_categorical(y_test, n_classes)

    x_test = dataset.X_test.astype("float32")

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    is_roi = hasattr(dataset, "pos_test") and dataset.pos_test is not None
    is_gesture = (cfg.mode in ("fwdPass", "hybrid")) and ("gesture" in cfg.dataset)

    if is_gesture:
        from test_utils import eval_model as gesture_eval
        x_input = [x_test, dataset.pos_test] if is_roi else x_test
        score = gesture_eval(model, x_input, y_test)
    elif is_roi:
        score = model.evaluate([x_test, dataset.pos_test.astype("float32")], y_test, verbose=2)
    else:
        score = model.evaluate(x_test, y_test, verbose=2)

    loss_val, acc_val = float(score[0]), float(score[1])
    print(f"[Saved model evaluation] loss={loss_val:.4f}  accuracy={acc_val:.4f}")
    return loss_val, acc_val


def _find_experiment_dirs(root: Path) -> "list[Path]":
    """
    Discover experiment directories under a parent folder.

    An "experiment directory" is any directory (at any depth under ``root``,
    ``root`` itself included) that directly contains at least one SLURM
    ``.out`` file. Directories are returned sorted, and directories without a
    ``config.yaml`` are kept in the list but will be skipped later with a
    warning (so the user gets visibility into what was found vs. usable).
    """
    out_files = sorted(root.rglob("*.out"))
    seen = []
    for f in out_files:
        parent = f.parent
        if parent not in seen:
            seen.append(parent)
    return seen


def run_single_experiment(experiment: Path, args, mode: str) -> dict:
    """
    Run the evaluate/retrain pipeline on a single experiment directory,
    according to the chosen selection/activation ``mode``:

      native_relu         - best iteration among natively-relu ones, retrain.
      force_relu_retrain   - best iteration overall, force activation='relu', retrain.
      force_relu_infer     - best iteration overall, force saved model's activation
                             to 'relu', inference only (no retrain).

    Returns a small summary dict; raises on unrecoverable errors (missing
    config.yaml, dataset load failure, etc.) so the caller can decide how to
    handle batch failures.
    """
    assert mode in SELECTION_MODES, f"Unknown selection mode: {mode}"
    require_relu = (mode == "native_relu")
    do_retrain = mode in ("native_relu", "force_relu_retrain")
    load_saved_model = mode in ("force_relu_retrain", "force_relu_infer")

    config_path = experiment / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {experiment}")

    # ── 0. Activate the experiment config ────────────────────────────────────
    from exp_config import set_active_config, load_cfg, reload_cfg

    set_active_config(config_path)
    cfg = load_cfg(force=True)

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
    print(f"Experiment       : {experiment}")
    print(f"Dataset          : {cfg.dataset}")
    print(f"Backend          : {cfg.backend}")
    print(f"Epochs           : {cfg.epochs}")
    print(f"Selection mode   : {mode}")
    print(f"Will retrain     : {do_retrain}")
    print(f"{'='*60}\n")

    # ── 1. Load the dataset ───────────────────────────────────────────────────
    roi_frame_size = 32
    print(f"\n[1] Loading dataset '{cfg.dataset}'" +
          (f" (frame_size={roi_frame_size})" if "roigesture" in cfg.dataset.lower() else "") +
          "...")
    dataset = _load_dataset(cfg.dataset, frame_size=roi_frame_size)
    dataset.data_as_float32()
    print(
        f"[1] Dataset loaded: "
        f"{dataset.X_train.shape[0]} train / {dataset.X_test.shape[0]} test samples."
    )

    # ── 2. Find the best iteration, subject to the selection-mode policy ─────
    print(f"\n[2] Finding best iteration (selection-mode='{mode}', "
          f"require_relu={require_relu})...")
    out_result = _parse_best_from_out(experiment, require_relu=require_relu)
    if out_result is not None:
        best_params, layer_x_block, best_idx, best_acc, best_score = out_result
    else:
        print("[2] No matching .out file found — falling back to algorithm_logs/")
        algo_logs = experiment / "algorithm_logs"
        best_idx, best_params, metric_name, metric_value = _find_best_iteration_fallback(
            algo_logs, require_relu=require_relu
        )
        layer_x_block = _find_layer_x_block(experiment, best_idx)
        best_acc = metric_value if metric_name == "acc" else None
        best_score = metric_value if metric_name == "score" else None
    print(f"[2] layer_x_block={layer_x_block}")

    best_params = dict(best_params)
    original_activation = best_params.get("activation")
    if mode == "force_relu_retrain":
        best_params["activation"] = "relu"
        print(f"[2] Activation forced: '{original_activation}' -> 'relu' (will be retrained).")
    elif mode == "native_relu":
        best_params["activation"] = "relu"  # already relu by construction; kept explicit
    # force_relu_infer: best_params activation left as originally found; the
    # FORCED relu is applied only to the saved model's layers for inference.

    saved_loss = best_score
    saved_acc = best_acc

    # ── 3. Load the saved Keras model (only when relevant for this mode) ─────
    saved_model = None
    if load_saved_model:
        keras_model_path = experiment / "Model" / "best-model.keras"
        if not keras_model_path.exists():
            print(
                f"[WARNING] {keras_model_path} not found. "
                "Skipping pre-trained model evaluation."
            )
        else:
            print(f"\n[3] Loading model from: {keras_model_path}")
            from test_utils import load_keras_model
            saved_model = load_keras_model(str(keras_model_path))
            print("[3] Model loaded.")
    else:
        print(
            "\n[3] Skipped: in 'native_relu' mode the on-disk best-model.keras "
            "corresponds to the OVERALL best iteration, which may differ from "
            "the selected natively-relu iteration — so it is not evaluated."
        )

    # ── 4. Evaluate the saved model (only when relevant for this mode) ───────
    if saved_model is not None:
        if mode == "force_relu_infer":
            _force_relu_activations(saved_model)
        saved_model.summary()
        print("\n[4] Evaluating saved model (best-model.keras)...")
        saved_loss, saved_acc = _eval_keras_model(saved_model, dataset, cfg)
    else:
        print("\n[4] No saved model evaluated for this mode.")

    # ── Fix cfg.name to an absolute path (needed by training()) ──────────────
    if str(experiment) != cfg.name:
        import yaml
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)
        raw["name"] = str(experiment)
        with open(config_path, "w") as f:
            yaml.safe_dump(raw, f, sort_keys=False, allow_unicode=True)
        cfg = reload_cfg()
        print(f"[Config] Updated 'name' to absolute path: {experiment}")

    summary = {
        "experiment": experiment,
        "mode": mode,
        "saved_loss": saved_loss,
        "saved_acc": saved_acc,
        "best_idx": best_idx,
        "layer_x_block": layer_x_block,
        "best_params": best_params,
        "retrain_loss": None,
        "retrain_acc": None,
    }

    if not do_retrain:
        print(f"\n{'='*60}")
        print(f"SUMMARY  (mode='{mode}' — inference only)")
        if saved_acc is not None:
            print(f"  Saved/forced-relu model  →  loss={saved_loss:.4f}  acc={saved_acc:.4f}")
        print(f"  Best iteration          : {best_idx}")
        print(f"  layer_x_block           : {layer_x_block}")
        print(f"  Hyperparameters         : {best_params}")
        print(f"{'='*60}\n")
        return summary

    # ── 5. Rebuild the architecture with the configured backend ──────────────
    print(f"\n[5] Rebuilding model with backend='{cfg.backend}'...")

    if cfg.backend == "tf":
        from tensorflow_implementation import module_backend, neural_network
        from tensorflow.keras import backend as K
        K.clear_session()
    elif cfg.backend == "torch":
        from pytorch_implementation import module_backend, neural_network
    else:
        raise ValueError(f"Unsupported backend: {cfg.backend}")

    backend_instance = module_backend.ModuleBackend()
    nn = neural_network.NeuralNetwork(
        backend=backend_instance,
        dataset=dataset,
        da=best_params.get("data_augmentation", False),
        reg=best_params.get("reg_l2", False),
        residual=False,
    )

    nn.build_network(best_params, layer_x_block=layer_x_block)
    print(f"[5] Model built (backend={cfg.backend}).")

    # ── 6. Retrain from scratch and evaluate ─────────────────────────────────
    print(f"\n[6] Retraining for {cfg.epochs} epoch(s)...")
    score, history, trained_model = nn.training(best_params)

    retrain_loss = float(score[0])
    retrain_acc = float(score[1])
    print(f"\n[6] Retrain results: loss={retrain_loss:.4f}  accuracy={retrain_acc:.4f}")

    summary["retrain_loss"] = retrain_loss
    summary["retrain_acc"] = retrain_acc

    print(f"\n{'='*60}")
    print(f"SUMMARY  (mode='{mode}')")
    if saved_acc is not None:
        print(f"  Reference (pre-retrain)  →  loss={saved_loss:.4f}  acc={saved_acc:.4f}")
    print(f"  Retrained model          →  loss={retrain_loss:.4f}  acc={retrain_acc:.4f}")
    print(f"  Best iteration           : {best_idx}")
    print(f"  layer_x_block            : {layer_x_block}")
    print(f"  Hyperparameters          : {best_params}")
    print(f"{'='*60}\n")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and optionally retrain a Symbolic DNN Tuner experiment."
    )
    parser.add_argument(
        "--experiment", required=True,
        help="Path to an experiment directory (must contain config.yaml), OR a "
             "parent directory containing one or more experiment subfolders. "
             "In the latter case (batch mode), every subfolder (at any depth) "
             "that contains at least one SLURM '.out' file is analyzed in "
             "REPORT-ONLY mode: no dataset/model is ever loaded and nothing "
             "is ever retrained — only the best-overall and best-native-relu "
             "iterations are found and printed for each."
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
        "--selection-mode", type=str, default=None, choices=list(SELECTION_MODES),
        help=(
            "Policy for choosing the iteration and handling its activation: "
            "'native_relu' (best iteration already trained with relu, then "
            "retrain it), 'force_relu_retrain' (best iteration overall, force "
            "activation to relu, then retrain), 'force_relu_infer' (best "
            "iteration overall, force the saved model's activation to relu, "
            "inference only). If omitted, you will be prompted interactively."
        ),
    )
    args = parser.parse_args()

    root = Path(args.experiment).expanduser().resolve()
    if not root.is_dir():
        print(f"[ERROR] Directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    # ── Single experiment: full pipeline (mode prompt, eval, optional retrain) ──
    if (root / "config.yaml").exists():
        mode = args.selection_mode or _prompt_selection_mode()
        try:
            run_single_experiment(root, args, mode)
        except Exception as exc:
            print(f"[ERROR] {root}: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    # ── Batch (parent folder): REPORT ONLY ───────────────────────────────────
    # When a whole parent folder is passed, the script NEVER loads a dataset,
    # NEVER loads/evaluates the saved Keras model, and NEVER retrains anything.
    # For every experiment subfolder found it only looks up and prints two
    # results: the best iteration overall, and the best iteration among those
    # natively trained with activation='relu'.
    print(f"[Batch] '{root}' has no config.yaml directly — scanning for "
          f"experiment subfolders (any dir containing a '*.out' file)...")
    print("[Batch] Report-only mode: nessun dataset/modello verra' caricato o "
          "riaddestrato; verranno solo stampati, per ogni esperimento, il "
          "miglior modello assoluto e il miglior modello con attivazione "
          "'relu' nativa.")
    experiment_dirs = _find_experiment_dirs(root)

    if not experiment_dirs:
        print(f"[ERROR] No subfolder with a '.out' file found under {root}", file=sys.stderr)
        sys.exit(1)

    print(f"[Batch] Found {len(experiment_dirs)} candidate experiment folder(s):")
    for d in experiment_dirs:
        print(f"  - {d}")

    results = []
    skipped = []
    for i, exp_dir in enumerate(experiment_dirs, start=1):
        print(f"\n{'#'*70}")
        print(f"# Batch [{i}/{len(experiment_dirs)}]  {exp_dir}")
        print(f"{'#'*70}")

        if not (exp_dir / "config.yaml").exists():
            msg = f"skipped — no config.yaml in {exp_dir}"
            print(f"[WARNING] {msg}")
            skipped.append((exp_dir, msg))
            continue

        info = _find_best_overall_and_relu(exp_dir)
        _print_candidate("Miglior modello assoluto", info.get("overall"), info.get("overall_error"))
        _print_candidate("Miglior modello (relu nativa)", info.get("relu"), info.get("relu_error"))

        results.append({"experiment": exp_dir, **info})

    # ── Batch-wide summary ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"BATCH SUMMARY  —  {len(results)} esperimenti analizzati, "
          f"{len(skipped)} saltati (su {len(experiment_dirs)} totali)")
    print(f"{'='*70}")
    for r in results:
        overall = r.get("overall")
        relu = r.get("relu")
        overall_str = (f"{overall['metric_name']}={overall['metric_value']:.4f}"
                        if overall else "n/d")
        relu_str = (f"{relu['metric_name']}={relu['metric_value']:.4f}"
                    if relu else "n/d")
        print(f"  {r['experiment']}  —  assoluto: {overall_str}  |  relu: {relu_str}")
    for d, msg in skipped:
        print(f"  [SKIPPED] {d}  ->  {msg}")
    print(f"{'='*70}\n")

    print("Esperimento con lo score/accuracy migliore, per categoria:")
    _print_batch_best_group(results, "overall", "Miglior modello assoluto")
    _print_batch_best_group(results, "relu", "Miglior modello (relu nativa)")
    print()

    if skipped and not results:
        sys.exit(1)


if __name__ == "__main__":
    main()