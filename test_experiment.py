"""
test_experiment.py  –  Evaluate a pre-trained experiment and optionally retrain it.

Usage:
    # Single experiment (must contain config.yaml directly)
    python test_experiment.py --experiment <path/to/experiment_dir> \
        [--selection-mode MODE] [--activation relu|selu]

    # Batch mode (REPORT ONLY): <path> is a parent folder containing one or
    # more experiment subfolders. Any subfolder (at any depth) that contains
    # at least one SLURM ".out" file is treated as an experiment directory.
    # In this mode the script NEVER loads a dataset, NEVER loads/evaluates the
    # saved Keras model, and NEVER retrains anything — for each experiment it
    # only looks up and prints the best iteration overall and the best
    # iteration among those natively trained with the target activation.
    # --selection-mode/--epochs/--backend are ignored in this mode
    # (--activation is still used to decide which "native" activation to
    # report on).
    python test_experiment.py --experiment <path/to/parent_dir> [--activation relu|selu]

Selection / activation policy (--selection-mode), chosen once per run (asked
interactively if not passed on the command line). The TARGET activation
(--activation, 'relu' or 'selu') is also chosen once per run:

  native_activation        Find the best iteration AMONG THOSE that already
                            used the TARGET activation natively (i.e. it was
                            the value chosen by the hyper-parameter search,
                            not forced), and retrain that architecture from
                            scratch.

  force_activation_retrain Find the best iteration overall (regardless of its
                            original activation), force its activation to the
                            TARGET activation, and retrain that architecture
                            from scratch.

  force_activation_infer   Find the best iteration overall (regardless of its
                            original activation), load the corresponding saved
                            Keras model (Model/best-model.keras), force its
                            layers' activation to the TARGET activation at
                            inference time, and only evaluate it (no
                            retraining).

  auto_activation_infer    No --activation needed. Compare the best LOGGED
                            accuracy of the relu iterations vs the selu ones,
                            pick the winner automatically, then evaluate
                            Model/best-model.keras on the TEST SET ONLY (no
                            retraining). For gesture / roigesture_* the test
                            split is framed with the config's delta_t.
                            best-model.keras is evaluated as-is; pass
                            --force-winner-activation to patch it to the winner
                            when they differ. Optional: --delta-t <us>.

  auto_activation_retrain  No --activation needed and NO saved model needed.
                            Same relu-vs-selu choice by logged accuracy, then
                            REBUILD that iteration's architecture from the logs
                            (hyper-neural.txt + layer_x_block), retrain it from
                            scratch and test it. gesture / roigesture_* use the
                            config's delta_t. Optional: --delta-t <us>,
                            --epochs, --backend.

Steps (applied to each experiment directory):
  1) Load  the dataset described in the experiment's config.yaml
  2) Find  the best iteration (subject to the chosen selection-mode policy)
           from the SLURM .out log, or — as a fallback — from
           algorithm_logs/hyper-neural.txt using score_report.txt as the
           index (acc_report.txt as fallback)
  3) Load  Model/best-model.keras                (modes: force_activation_retrain, force_activation_infer)
  4) Test  the saved model  (accuracy and loss)   (modes: force_activation_retrain, force_activation_infer)
  5) Rebuild the same architecture with the configured backend (tf or torch)  (modes: native_activation, force_activation_retrain)
  6) Retrain and test                                                         (modes: native_activation, force_activation_retrain)
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from pathlib import Path


SELECTION_MODES = (
    "native_activation",
    "force_activation_retrain",
    "force_activation_infer",
    "auto_activation_infer",
    "auto_activation_retrain",
)
SUPPORTED_ACTIVATIONS = ("relu", "selu")

SELECTION_MODE_PROMPTS = {
    "1": ("native_activation",
          "Trova il miglior modello che ha GIA' l'attivazione scelta fin "
          "dall'origine, e riaddestra quello."),
    "2": ("force_activation_retrain",
          "Prendi il modello migliore in assoluto, imposta l'attivazione "
          "scelta e riaddestralo da zero."),
    "3": ("force_activation_infer",
          "Prendi il modello migliore in assoluto, imposta l'attivazione "
          "scelta e fai SOLO inferenza (nessun riaddestramento)."),
    "4": ("auto_activation_infer",
          "Confronta l'accuratezza (dai log) del miglior modello relu e del "
          "miglior modello selu, scegli automaticamente il vincitore e "
          "valuta best-model.keras sul SOLO test set (nessun riaddestramento)."),
    "5": ("auto_activation_retrain",
          "Come la 4 per la scelta relu/selu, ma RICOSTRUISCE l'architettura "
          "vincente dai log, la riaddestra da zero e la testa (non serve "
          "best-model.keras)."),
}


def _prompt_selection_mode() -> str:
    """Ask the user, once per run, which selection/activation policy to use."""
    print("\nSeleziona la modalita' di scelta del modello / attivazione:")
    for key, (_, desc) in SELECTION_MODE_PROMPTS.items():
        print(f"  {key}) {desc}")
    while True:
        choice = input("Scelta [1/2/3/4/5]: ").strip()
        if choice in SELECTION_MODE_PROMPTS:
            mode = SELECTION_MODE_PROMPTS[choice][0]
            print(f"[Selection mode] '{mode}' selezionato.\n")
            return mode
        print("Scelta non valida, riprova.")


def _prompt_activation() -> str:
    """Ask the user, once per run, which target activation to use."""
    print("\nSeleziona l'attivazione target:")
    for i, act in enumerate(SUPPORTED_ACTIVATIONS, start=1):
        print(f"  {i}) {act}")
    while True:
        choice = input(f"Scelta [1-{len(SUPPORTED_ACTIVATIONS)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(SUPPORTED_ACTIVATIONS):
            activation = SUPPORTED_ACTIVATIONS[int(choice) - 1]
            print(f"[Activation] '{activation}' selezionata.\n")
            return activation
        # Also accept the activation name typed directly
        if choice.lower() in SUPPORTED_ACTIVATIONS:
            activation = choice.lower()
            print(f"[Activation] '{activation}' selezionata.\n")
            return activation
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


def _load_dataset(dataset_name: str, frame_size: int = 32, test_only: bool = False):
    """Instantiate and load the correct TunerDataset for the given dataset name.

    ``test_only`` is honoured only for the DVSGesture datasets (gesture /
    roigesture_*); the other loaders always build their full split.
    """
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
        ds.load_gesture(test_only=test_only)
    elif "roigesture" in name:
        ds.load_roi_gesture(frame_size=frame_size, test_only=test_only)
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
    require_activation: "str | None" = None,
    rank_by: str = "auto",
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

    If ``require_activation`` is set (e.g. 'relu' or 'selu'), only iterations
    whose original hyperparameters already used that activation are
    considered candidates (this implements the 'native_activation' selection
    mode). If ``require_activation`` is None, no activation filter is applied.

    ``rank_by`` forces the ranking metric: 'acc' always maximises ACCURACY,
    'auto' (default) prefers SCORE when available.

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

            # Rank: prefer SCORE (lower = better), fall back to ACCURACY (higher = better).
            # rank_by='acc' forces accuracy ranking regardless of SCORE availability.
            has_score = (rank_by != "acc") and any(it["score"] is not None for it in iterations)
            valid = [
                it for it in iterations
                if (require_activation is None or it["params"].get("activation") == require_activation)
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
    algo_logs: Path, require_activation: "str | None" = None, prefer_acc: bool = False
) -> "tuple[int, dict, str, float]":
    """
    Fallback selection (used when no .out file is available): rank iterations
    using score_report.txt (preferred, lower = better) or acc_report.txt
    (higher = better), reading the corresponding hyperparameters from
    hyper-neural.txt (same line index).

    If ``require_activation`` is set (e.g. 'relu' or 'selu'), only iterations
    whose hyperparameters already used that activation are considered (the
    'native_activation' selection mode). If None, no filter is applied.

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

    if prefer_acc and acc_path.exists():
        raw_lines = acc_path.read_text().splitlines()
        use_score = False
        source_name = "acc_report.txt"
    elif score_path.exists():
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
        if require_activation is not None and all_params[i].get("activation") != require_activation:
            continue
        candidates.append(i)

    if not candidates:
        scope = f" con activation='{require_activation}'" if require_activation else ""
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


def _find_best_overall_and_activation(experiment: Path, activation: str) -> dict:
    """
    For a single experiment directory, find BOTH:
      - the best iteration overall (any activation)
      - the best iteration among those natively trained with the TARGET
        activation (e.g. 'relu' or 'selu')

    without touching the dataset or any saved Keras model — this is used by
    batch mode, which only reports these two results and never loads data,
    loads a saved model, evaluates it, or retrains anything.

    Returns a dict:
        {
          "overall":    {"idx", "params", "layer_x_block", "metric_name", "metric_value"} | None,
          "overall_error": str            # present only if "overall" is None
          "activation": {"idx", "params", "layer_x_block", "metric_name", "metric_value"} | None,
          "activation_error": str          # present only if "activation" is None
        }
    """
    result: dict = {"overall": None, "activation": None}

    out_overall = _parse_best_from_out(experiment, require_activation=None)
    out_native = _parse_best_from_out(experiment, require_activation=activation)

    def _pack_from_out(out_result):
        params, lxb, idx, acc, score = out_result
        metric_name, metric_value = ("score", score) if score is not None else ("acc", acc)
        return {"idx": idx, "params": params, "layer_x_block": lxb,
                "metric_name": metric_name, "metric_value": metric_value}

    if out_overall is not None:
        result["overall"] = _pack_from_out(out_overall)
    if out_native is not None:
        result["activation"] = _pack_from_out(out_native)

    if out_overall is None or out_native is None:
        algo_logs = experiment / "algorithm_logs"
        if out_overall is None:
            try:
                idx, params, metric_name, metric_value = _find_best_iteration_fallback(
                    algo_logs, require_activation=None
                )
                lxb = _find_layer_x_block(experiment, idx)
                result["overall"] = {"idx": idx, "params": params, "layer_x_block": lxb,
                                      "metric_name": metric_name, "metric_value": metric_value}
            except Exception as exc:
                result["overall_error"] = str(exc)
        if out_native is None:
            try:
                idx, params, metric_name, metric_value = _find_best_iteration_fallback(
                    algo_logs, require_activation=activation
                )
                lxb = _find_layer_x_block(experiment, idx)
                result["activation"] = {"idx": idx, "params": params, "layer_x_block": lxb,
                                         "metric_name": metric_name, "metric_value": metric_value}
            except Exception as exc:
                result["activation_error"] = str(exc)

    return result


def _print_candidate(label: str, cand: "dict | None", error: "str | None" = None) -> None:
    """Pretty-print a single best-iteration candidate (or the reason it's missing)."""
    if cand is None:
        msg = error or "nessuna iterazione trovata"
        print(f"  {label:<32}: NON TROVATO — {msg}")
        return
    print(
        f"  {label:<32}: iterazione={cand['idx']}  layer_x_block={cand['layer_x_block']}  "
        f"{cand['metric_name']}={cand['metric_value']:.4f}"
    )
    print(f"    hyperparams: {cand['params']}")


def _print_batch_best_group(results: "list[dict]", key: str, group_label: str) -> None:
    """
    Across all experiments in a batch, print which one obtained the best value
    for ``key`` ("overall" or "activation"), separately for score-ranked and
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


def _force_activation(model, activation: str) -> None:
    """
    Monkey-patch every layer's stored activation function to the TARGET
    activation ('relu' or 'selu'), in place.

    This is used only by the 'force_activation_infer' selection mode: it
    swaps the activation of an ALREADY TRAINED model at inference time
    (weights are not retrained/recompiled), purely to measure how the saved
    checkpoint behaves if its activation had been the target one.
    """
    import tensorflow as tf

    if activation not in SUPPORTED_ACTIVATIONS:
        raise ValueError(
            f"Unsupported activation '{activation}'. Supported: {SUPPORTED_ACTIVATIONS}"
        )

    changed = 0
    for layer in model.layers:
        if hasattr(layer, "activation") and layer.activation is not None:
            layer.activation = tf.keras.activations.get(activation)
            changed += 1
    print(f"[Force {activation}] Patched activation on {changed} layer(s) of the saved model.")


def _model_activation(model) -> "str | None":
    """Best-effort read of a saved model's hidden activation ('relu'/'selu'/...)."""
    from collections import Counter
    names = Counter()
    for layer in model.layers:
        act = getattr(layer, "activation", None)
        name = getattr(act, "__name__", None)
        if name and name not in ("linear", "softmax", "sigmoid"):
            names[name] += 1
    return names.most_common(1)[0][0] if names else None


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
    import numpy as np
    import tensorflow as tf

    n_classes = dataset.n_classes

    y_test = np.asarray(dataset.Y_test)
    if y_test.ndim == 1:
        y_test = tf.keras.utils.to_categorical(y_test, n_classes)

    x_test = dataset.X_test
    if getattr(x_test, "dtype", None) != object:
        x_test = x_test.astype("float32")

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


# ---------------------------------------------------------------------------
# Automatic relu-vs-selu selection (by accuracy, from the training logs)
# ---------------------------------------------------------------------------

def _best_acc_iteration(experiment: Path, activation: str) -> "dict | None":
    """
    Best iteration (by ACCURACY, from the training logs) among those whose
    hyperparameters natively used ``activation``. Tries the SLURM .out first,
    then falls back to algorithm_logs/acc_report.txt + hyper-neural.txt.

    Returns {"idx", "acc", "params", "layer_x_block"} or None.
    """
    out = _parse_best_from_out(experiment, require_activation=activation, rank_by="acc")
    if out is not None:
        params, lxb, idx, acc, _score = out
        return {"idx": idx, "acc": acc, "params": params, "layer_x_block": lxb}

    try:
        idx, params, metric_name, metric_value = _find_best_iteration_fallback(
            experiment / "algorithm_logs", require_activation=activation, prefer_acc=True
        )
    except Exception as exc:
        print(f"[auto] nessun candidato '{activation}': {exc}")
        return None
    if metric_name != "acc":
        print(f"[auto] '{activation}': metrica disponibile '{metric_name}', non accuracy — ignorato.")
        return None
    lxb = _find_layer_x_block(experiment, idx)
    return {"idx": idx, "acc": metric_value, "params": params, "layer_x_block": lxb}


def _auto_select_activation(experiment: Path) -> "tuple[str, dict]":
    """
    Pick relu vs selu automatically as the activation whose best iteration has
    the highest logged accuracy.

    Returns (winner_activation, {act: candidate|None}).
    """
    cands = {a: _best_acc_iteration(experiment, a) for a in SUPPORTED_ACTIVATIONS}
    scored = {a: c for a, c in cands.items() if c is not None and c["acc"] is not None}
    if not scored:
        raise ValueError(
            "Impossibile scegliere l'attivazione: nessuna iterazione con accuracy "
            f"trovata per relu/selu in {experiment}."
        )
    winner = max(scored, key=lambda a: scored[a]["acc"])
    return winner, cands


def _run_auto_activation_infer(experiment: Path, args) -> dict:
    """
    'auto_activation_infer': choose relu/selu by best logged accuracy, then
    evaluate Model/best-model.keras on the TEST SET ONLY (no retraining),
    forcing the winning activation. Works for gesture and roigesture_*.
    """
    config_path = experiment / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {experiment}")

    from exp_config import set_active_config, load_cfg, reload_cfg
    set_active_config(config_path)
    cfg = load_cfg(force=True)

    # Optional Δt override from the command line.
    if getattr(args, "delta_t", None) is not None:
        import yaml
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)
        cfg["delta_t"] = int(args.delta_t)
        # with open(config_path, "w") as f:
        #     yaml.safe_dump(raw, f, sort_keys=False, allow_unicode=True)
        # cfg = reload_cfg()

    winner, cands = _auto_select_activation(experiment)

    print(f"\n{'='*60}")
    print(f"Experiment       : {experiment}")
    print(f"Dataset          : {cfg.dataset}")
    print(f"Mode             : {cfg.mode}   delta_t: {cfg.delta_t}")
    print(f"Selection mode   : auto_activation_infer  (test set only)")
    print(f"{'='*60}")
    for a in SUPPORTED_ACTIVATIONS:
        c = cands.get(a)
        if c is None:
            print(f"  {a:<4} : nessun candidato con accuracy nei log")
        else:
            print(f"  {a:<4} : iter={c['idx']}  acc(log)={c['acc']:.4f}  "
                  f"layer_x_block={c['layer_x_block']}")
    print(f"  -> attivazione vincente: '{winner}'")
    print(f"{'='*60}\n")

    roi_frame_size = 32
    if "roigesture" in cfg.dataset.lower():
        roi_frame_size = _detect_roi_frame_size(experiment)
    print(f"[1] Loading TEST split of '{cfg.dataset}'" +
          (f" (frame_size={roi_frame_size})" if "roigesture" in cfg.dataset.lower() else "") + "...")
    dataset = _load_dataset(cfg.dataset, frame_size=roi_frame_size, test_only=True)
    dataset.data_as_float32()
    print(f"[1] {len(dataset.X_test)} test samples.")

    keras_model_path = experiment / "Model" / "best-model.keras"
    if not keras_model_path.exists():
        raise FileNotFoundError(f"{keras_model_path} not found — nothing to evaluate.")
    print(f"\n[2] Loading model: {keras_model_path}")
    from test_utils import load_keras_model
    saved_model = load_keras_model(str(keras_model_path))
    native_act = _model_activation(saved_model)
    print(f"[2] Saved model native activation: {native_act or 'sconosciuta'}")
    saved_model.summary()

    forced = getattr(args, "force_winner_activation", False)
    if forced and native_act != winner:
        print(f"\n[!] best-model.keras è stato addestrato con '{native_act}', "
              f"non '{winner}'. Con --force-winner-activation forzo '{winner}' "
              f"sui pesi esistenti: il risultato NON riflette le vere prestazioni "
              f"del miglior modello '{winner}' (che non è salvato su disco).")
        _force_activation(saved_model, winner)
        eval_act = winner
    else:
        if native_act != winner:
            print(f"\n[i] L'attivazione vincente dai log è '{winner}', ma l'unico "
                  f"modello salvato (best-model.keras) usa '{native_act}'. "
                  f"Valuto il modello salvato COSÌ COM'È. Usa "
                  f"--force-winner-activation per forzare '{winner}'.")
        eval_act = native_act

    print(f"\n[3] Evaluating best-model.keras on the TEST set (activation='{eval_act}')...")
    loss_val, acc_val = _eval_keras_model(saved_model, dataset, cfg)

    print(f"\n{'='*60}")
    print(f"SUMMARY  (auto_activation_infer)")
    for a in SUPPORTED_ACTIVATIONS:
        c = cands.get(a)
        print(f"  best {a:<4} (log acc)     : {c['acc']:.4f}" if c else
              f"  best {a:<4} (log acc)     : n/d")
    print(f"  Attivazione vincente     : {winner}")
    print(f"  best-model.keras attiv.   : {native_act or 'sconosciuta'}  "
          f"(valutato come '{eval_act}')")
    print(f"  Test set  → loss={loss_val:.4f}  acc={acc_val:.4f}")
    print(f"{'='*60}\n")

    return {
        "experiment": experiment,
        "mode": "auto_activation_infer",
        "activation": winner,
        "eval_activation": eval_act,
        "native_activation": native_act,
        "candidates": {a: (c["acc"] if c else None) for a, c in cands.items()},
        "test_loss": loss_val,
        "test_acc": acc_val,
    }


def run_single_experiment(experiment: Path, args, mode: str, activation: str) -> dict:
    """
    Run the evaluate/retrain pipeline on a single experiment directory,
    according to the chosen selection/activation ``mode`` and TARGET
    ``activation`` ('relu' or 'selu'):

      native_activation        - best iteration among natively-target-activation
                                  ones, retrain.
      force_activation_retrain - best iteration overall, force activation to
                                  the target, retrain.
      force_activation_infer   - best iteration overall, force saved model's
                                  activation to the target, inference only
                                  (no retrain).

    Returns a small summary dict; raises on unrecoverable errors (missing
    config.yaml, dataset load failure, etc.) so the caller can decide how to
    handle batch failures.
    """
    assert mode in SELECTION_MODES, f"Unknown selection mode: {mode}"

    if mode == "auto_activation_infer":
        # relu/selu chosen by the script (best logged accuracy); test set only.
        return _run_auto_activation_infer(experiment, args)

    config_path = experiment / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {experiment}")

    # 'auto_activation_retrain': pick relu/selu automatically (best logged
    # accuracy), then behave like 'native_activation' for that winner.
    auto_cands = None
    auto_pick = None
    if mode == "auto_activation_retrain":
        activation, auto_cands = _auto_select_activation(experiment)
        auto_pick = auto_cands[activation]
        print(f"[auto] attivazione vincente: '{activation}' "
              f"(acc log={auto_pick['acc']:.4f}, iter={auto_pick['idx']})")

    assert activation in SUPPORTED_ACTIVATIONS, f"Unsupported activation: {activation}"
    native_like = mode in ("native_activation", "auto_activation_retrain")
    require_activation = activation if native_like else None
    do_retrain = mode in ("native_activation", "force_activation_retrain", "auto_activation_retrain")
    load_saved_model = mode in ("force_activation_retrain", "force_activation_infer")

    # ── 0. Activate the experiment config ────────────────────────────────────
    from exp_config import set_active_config, load_cfg, reload_cfg

    set_active_config(config_path)
    cfg = load_cfg(force=True)

    _delta_t = getattr(args, "delta_t", None)
    _train_framing = getattr(args, "train_framing", None)
    if (args.epochs is not None or args.backend is not None or _delta_t is not None
            or _train_framing is not None):
        import yaml

        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)
        if args.epochs is not None:
            raw["epochs"] = args.epochs
        if args.backend is not None:
            raw["backend"] = args.backend
        if _delta_t is not None:
            raw["delta_t"] = int(_delta_t)
        if _train_framing is not None:
            raw["train_framing"] = _train_framing
        with open(config_path, "w") as f:
            yaml.safe_dump(raw, f, sort_keys=False, allow_unicode=True)
        cfg = reload_cfg()

    print(f"\n{'='*60}")
    print(f"Experiment       : {experiment}")
    print(f"Dataset          : {cfg.dataset}")
    print(f"Backend          : {cfg.backend}")
    print(f"Epochs           : {cfg.epochs}")
    if cfg.mode == "fwdPass":
        print(f"delta_t          : {cfg.delta_t}")
        if do_retrain:
            print(f"train_framing    : {cfg.train_framing}  "
                  f"({'fixed Δt (variable #frames)' if cfg.train_framing == 'delta_t' else f'fixed {cfg.frames} frames'})")
    print(f"Selection mode   : {mode}")
    print(f"Target activation: {activation}")
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
          f"require_activation={require_activation})...")
    if mode == "auto_activation_retrain":
        # Winner already found by accuracy in _auto_select_activation — reuse it.
        best_params = auto_pick["params"]
        layer_x_block = auto_pick["layer_x_block"]
        best_idx = auto_pick["idx"]
        best_acc, best_score = auto_pick["acc"], None
    else:
        out_result = _parse_best_from_out(experiment, require_activation=require_activation)
        if out_result is not None:
            best_params, layer_x_block, best_idx, best_acc, best_score = out_result
        else:
            print("[2] No matching .out file found — falling back to algorithm_logs/")
            algo_logs = experiment / "algorithm_logs"
            best_idx, best_params, metric_name, metric_value = _find_best_iteration_fallback(
                algo_logs, require_activation=require_activation
            )
            layer_x_block = _find_layer_x_block(experiment, best_idx)
            best_acc = metric_value if metric_name == "acc" else None
            best_score = metric_value if metric_name == "score" else None
    print(f"[2] layer_x_block={layer_x_block}")

    best_params = dict(best_params)
    original_activation = best_params.get("activation")
    if mode == "force_activation_retrain":
        best_params["activation"] = activation
        print(f"[2] Activation forced: '{original_activation}' -> '{activation}' (will be retrained).")
    elif native_like:
        best_params["activation"] = activation  # already this activation by construction; kept explicit
    # force_activation_infer: best_params activation left as originally found; the
    # FORCED activation is applied only to the saved model's layers for inference.

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
            f"\n[3] Skipped: in '{mode}' mode the on-disk best-model.keras "
            "corresponds to the OVERALL best iteration, which may differ from "
            "the selected natively-target-activation iteration — so it is not evaluated."
        )

    # ── 4. Evaluate the saved model (only when relevant for this mode) ───────
    if saved_model is not None:
        if mode == "force_activation_infer":
            _force_activation(saved_model, activation)
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
        "activation": activation,
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
        print(f"SUMMARY  (mode='{mode}', activation='{activation}' — inference only)")
        if saved_acc is not None:
            print(f"  Saved/forced model       →  loss={saved_loss:.4f}  acc={saved_acc:.4f}")
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
    nn.save_model(name="retrained-model.keras")

    retrain_loss = float(score[0])
    retrain_acc = float(score[1])
    print(f"\n[6] Retrain results: loss={retrain_loss:.4f}  accuracy={retrain_acc:.4f}")

    summary["retrain_loss"] = retrain_loss
    summary["retrain_acc"] = retrain_acc

    print(f"\n{'='*60}")
    print(f"SUMMARY  (mode='{mode}', activation='{activation}')")
    if saved_loss is not None and saved_acc is not None:
        print(f"  Reference (pre-retrain)  →  loss={saved_loss:.4f}  acc={saved_acc:.4f}")
    elif saved_acc is not None:
        print(f"  Reference (log accuracy) →  acc={saved_acc:.4f}")
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
             "is ever retrained — only the best-overall and best-native-target-"
             "activation iterations are found and printed for each."
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
            "'native_activation' (best iteration already trained with the target "
            "activation, then retrain it), 'force_activation_retrain' (best "
            "iteration overall, force activation to the target, then retrain), "
            "'force_activation_infer' (best iteration overall, force the saved "
            "model's activation to the target, inference only). If omitted, you "
            "will be prompted interactively."
        ),
    )
    parser.add_argument(
        "--activation", type=str, default=None, choices=list(SUPPORTED_ACTIVATIONS),
        help=(
            "Target activation function to use for the 'native'/'force' "
            "selection logic: 'relu' or 'selu'. If omitted, you will be "
            "prompted interactively (only when needed). Ignored by "
            "'auto_activation_infer', which picks it automatically."
        ),
    )
    parser.add_argument(
        "--delta-t", type=int, default=None, dest="delta_t",
        help=(
            "Override delta_t (microseconds per frame) in the experiment's "
            "config.yaml before (re)training/evaluating. Used by the "
            "'auto_activation_*' modes and any retrain mode."
        ),
    )
    parser.add_argument(
        "--train-framing", type=str, default=None, choices=["frames", "delta_t"], dest="train_framing",
        help=(
            "fwdPass gesture/roigesture retrain only: how to frame the TRAINING "
            "split. 'frames' (default) = fixed number of frames (config's "
            "'frames'), i.e. a variable inter-frame time across recordings. "
            "'delta_t' = fixed time window (config's delta_t), i.e. a variable "
            "number of frames per recording, matching the TEST framing. The "
            "test split is always framed with delta_t regardless of this flag. "
            "If omitted, uses train_framing from config.yaml (default 'frames')."
        ),
    )
    parser.add_argument(
        "--force-winner-activation", action="store_true", dest="force_winner_activation",
        help=(
            "auto_activation_infer only: if best-model.keras was trained with a "
            "different activation than the log-accuracy winner, patch its layers "
            "to the winner before evaluating (weights unchanged; result is only "
            "indicative)."
        ),
    )
    args = parser.parse_args()

    _AUTO_MODES = ("auto_activation_infer", "auto_activation_retrain")

    root = Path(args.experiment).expanduser().resolve()
    if not root.is_dir():
        print(f"[ERROR] Directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    # ── Single experiment: full pipeline (mode prompt, eval, optional retrain) ──
    if (root / "config.yaml").exists():
        mode = args.selection_mode or _prompt_selection_mode()
        # 'auto_activation_*' modes select relu/selu on their own.
        if mode in _AUTO_MODES:
            activation = args.activation or SUPPORTED_ACTIVATIONS[0]
        else:
            activation = args.activation or _prompt_activation()
        # try:
        run_single_experiment(root, args, mode, activation)
        # except Exception as exc:
        #     print(f"[ERROR] {root}: {exc}", file=sys.stderr)
        #     sys.exit(1)
        return

    # ── Batch + auto_activation_* : run per experiment (no report-only) ──────
    if args.selection_mode in _AUTO_MODES:
        experiment_dirs = [d for d in _find_experiment_dirs(root) if (d / "config.yaml").exists()]
        if not experiment_dirs:
            print(f"[ERROR] No experiment (config.yaml + .out) found under {root}", file=sys.stderr)
            sys.exit(1)
        rows = []
        for i, exp_dir in enumerate(experiment_dirs, start=1):
            print(f"\n{'#'*70}\n# [{i}/{len(experiment_dirs)}]  {exp_dir}\n{'#'*70}")
            try:
                if args.selection_mode == "auto_activation_infer":
                    rows.append(_run_auto_activation_infer(exp_dir, args))
                else:
                    rows.append(run_single_experiment(
                        exp_dir, args, args.selection_mode, SUPPORTED_ACTIVATIONS[0]
                    ))
            except Exception as exc:
                print(f"[ERROR] {exp_dir}: {exc}", file=sys.stderr)
        print(f"\n{'='*70}\nBATCH SUMMARY ({args.selection_mode})\n{'='*70}")
        for r in rows:
            if args.selection_mode == "auto_activation_infer":
                print(f"  {r['experiment']}  ->  vincente={r['activation']}  "
                      f"eval={r['eval_activation']}  test_acc={r['test_acc']:.4f}  "
                      f"test_loss={r['test_loss']:.4f}")
            else:
                print(f"  {r['experiment']}  ->  attiv={r['activation']}  "
                      f"retrain_acc={r.get('retrain_acc')}  retrain_loss={r.get('retrain_loss')}")
        return

    # ── Batch (parent folder): REPORT ONLY ───────────────────────────────────
    # When a whole parent folder is passed, the script NEVER loads a dataset,
    # NEVER loads/evaluates the saved Keras model, and NEVER retrains anything.
    # For every experiment subfolder found it only looks up and prints two
    # results: the best iteration overall, and the best iteration among those
    # natively trained with the target activation.
    activation = args.activation or _prompt_activation()

    print(f"[Batch] '{root}' has no config.yaml directly — scanning for "
          f"experiment subfolders (any dir containing a '*.out' file)...")
    print(f"[Batch] Report-only mode: nessun dataset/modello verra' caricato o "
          f"riaddestrato; verranno solo stampati, per ogni esperimento, il "
          f"miglior modello assoluto e il miglior modello con attivazione "
          f"'{activation}' nativa.")
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

        info = _find_best_overall_and_activation(exp_dir, activation)
        _print_candidate("Miglior modello assoluto", info.get("overall"), info.get("overall_error"))
        _print_candidate(f"Miglior modello ({activation} nativa)", info.get("activation"), info.get("activation_error"))

        results.append({"experiment": exp_dir, **info})

    # ── Batch-wide summary ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"BATCH SUMMARY  —  {len(results)} esperimenti analizzati, "
          f"{len(skipped)} saltati (su {len(experiment_dirs)} totali)  —  attivazione target: {activation}")
    print(f"{'='*70}")
    for r in results:
        overall = r.get("overall")
        native = r.get("activation")
        overall_str = (f"{overall['metric_name']}={overall['metric_value']:.4f}"
                        if overall else "n/d")
        native_str = (f"{native['metric_name']}={native['metric_value']:.4f}"
                      if native else "n/d")
        print(f"  {r['experiment']}  —  assoluto: {overall_str}  |  {activation}: {native_str}")
    for d, msg in skipped:
        print(f"  [SKIPPED] {d}  ->  {msg}")
    print(f"{'='*70}\n")

    print("Esperimento con lo score/accuracy migliore, per categoria:")
    _print_batch_best_group(results, "overall", "Miglior modello assoluto")
    _print_batch_best_group(results, "activation", f"Miglior modello ({activation} nativa)")
    print()

    if skipped and not results:
        sys.exit(1)


if __name__ == "__main__":
    main()