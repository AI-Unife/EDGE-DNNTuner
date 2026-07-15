"""
test_experiment.py  –  Valuta un esperimento già addestrato e lo ri-addestra.

Uso:
    python test_experiment.py --experiment <path/to/experiment_dir> [--epochs N] [--backend tf|torch]

Passi:
  1) Carica Model/best-model.keras
  2) Carica il dataset dal config.yaml dell'esperimento
  3) Testa il modello salvato (accuracy e loss)
  4) Estrae l'iterazione migliore da algorithm_logs/hyper-neural.txt
     usando score_report.txt (o acc_report.txt come fallback)
  5) Ricrea lo stesso modello con il backend del config (tf o torch)
  6) Ri-addestra e testa
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

def _load_dataset(dataset_name: str):
    """Istanzia e carica il TunerDataset corretto."""
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
        ds.load_roi_gesture()
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
            f"Dataset '{dataset_name}' non riconosciuto. "
            "Supportati: cifar10, cifar100, mnist, beans, light, gesture, "
            "roigesture_*, tinyimagenet, cca, cim."
        )
    return ds


def _find_best_iteration(algo_logs: Path) -> int:
    """Restituisce l'indice (0-based) dell'iterazione migliore."""
    score_path = algo_logs / "score_report.txt"
    acc_path = algo_logs / "acc_report.txt"

    if score_path.exists():
        values = [
            float(l.strip())
            for l in score_path.read_text().splitlines()
            if l.strip() and l.strip().lower() != "none"
        ]
        if not values:
            raise ValueError(f"score_report.txt è vuoto in {algo_logs}")
        best_idx = min(range(len(values)), key=lambda i: values[i])
        print(
            f"[Selezione] score_report.txt – iterazione migliore: {best_idx} "
            f"(score={values[best_idx]:.4f})"
        )
    elif acc_path.exists():
        values = [
            float(l.strip())
            for l in acc_path.read_text().splitlines()
            if l.strip() and l.strip().lower() != "none"
        ]
        if not values:
            raise ValueError(f"acc_report.txt è vuoto in {algo_logs}")
        best_idx = max(range(len(values)), key=lambda i: values[i])
        print(
            f"[Selezione] acc_report.txt – iterazione migliore: {best_idx} "
            f"(acc={values[best_idx]:.4f})"
        )
    else:
        raise FileNotFoundError(
            f"Nessun score_report.txt né acc_report.txt trovato in {algo_logs}"
        )

    return best_idx


def _load_best_params(algo_logs: Path, best_idx: int) -> dict:
    """Legge la riga best_idx da hyper-neural.txt e la converte in dict."""
    hyper_path = algo_logs / "hyper-neural.txt"
    if not hyper_path.exists():
        raise FileNotFoundError(f"hyper-neural.txt non trovato in {algo_logs}")

    lines = [l.strip() for l in hyper_path.read_text().splitlines() if l.strip()]
    if best_idx >= len(lines):
        raise IndexError(
            f"best_idx={best_idx} fuori range: hyper-neural.txt ha {len(lines)} righe."
        )
    params = ast.literal_eval(lines[best_idx])
    print(f"[Iperparametri] {params}")
    return params


def _find_layer_x_block(experiment: Path, best_idx: int) -> int:
    """
    Cerca layer_x_block nel file .out dell'esperimento (blocco dell'iterazione
    migliore). Ritorna 2 come fallback se non trovato.
    """
    # Cerca prima nella dir dell'esperimento, poi nella cwd
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
                        f"[layer_x_block] Trovato={val} in '{out_file.name}' "
                        f"(iterazione {best_idx})"
                    )
                    return val

    print("[layer_x_block] Non trovato nel file .out – uso default=2")
    return 2


def _eval_keras_model(model, dataset, cfg) -> tuple[float, float]:
    """
    Valuta un modello Keras caricato da disco su X_test / Y_test.
    Restituisce (loss, accuracy).
    """
    import numpy as np
    import tensorflow as tf

    n_classes = dataset.n_classes

    # Converti le label in one-hot se necessario
    y_test = dataset.Y_test
    if y_test.ndim == 1:
        y_test = tf.keras.utils.to_categorical(y_test, n_classes)

    x_test = dataset.X_test.astype("float32")

    # Determina la loss da usare in base all'output shape del modello
    out_shape = model.output_shape
    if isinstance(out_shape, list):
        # modello multi-output (es. ROI)
        loss = "categorical_crossentropy"
    else:
        loss = "categorical_crossentropy"

    model.compile(loss=loss, optimizer="adam", metrics=["accuracy"])

    is_roi = hasattr(dataset, "pos_test") and dataset.pos_test is not None
    is_gesture = (
        (cfg.mode in ("fwdPass", "hybrid")) and ("gesture" in cfg.dataset)
    )

    if is_gesture:
        from test_utils import eval_model as gesture_eval
        score = gesture_eval(model, x_test if not is_roi else [x_test, dataset.pos_test], y_test)
    elif is_roi:
        score = model.evaluate([x_test, dataset.pos_test.astype("float32")], y_test, verbose=2)
    else:
        score = model.evaluate(x_test, y_test, verbose=2)

    loss_val, acc_val = float(score[0]), float(score[1])
    print(f"[Valutazione modello salvato] loss={loss_val:.4f}  accuracy={acc_val:.4f}")
    return loss_val, acc_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Testa e ri-addestra un esperimento Symbolic DNN Tuner."
    )
    parser.add_argument(
        "--experiment", required=True,
        help="Percorso della directory dell'esperimento (contiene config.yaml)."
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Sovrascrive il numero di epoche per il ri-addestramento (opzionale)."
    )
    parser.add_argument(
        "--backend", type=str, default=None, choices=["tf", "torch"],
        help="Sovrascrive il backend (tf o torch) per il ri-addestramento (opzionale)."
    )
    args = parser.parse_args()

    experiment = Path(args.experiment).expanduser().resolve()
    if not experiment.is_dir():
        print(f"[ERRORE] Directory non trovata: {experiment}", file=sys.stderr)
        sys.exit(1)

    config_path = experiment / "config.yaml"
    if not config_path.exists():
        print(f"[ERRORE] config.yaml non trovato in {experiment}", file=sys.stderr)
        sys.exit(1)

    # ── 0. Imposta il config attivo ──────────────────────────────────────────
    from exp_config import set_active_config, load_cfg, reload_cfg

    set_active_config(config_path)
    cfg = load_cfg(force=True)

    # Applica eventuali override da CLI
    if args.epochs is not None or args.backend is not None:
        import yaml, os as _os

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
    print(f"Esperimento : {experiment.name}")
    print(f"Dataset     : {cfg.dataset}")
    print(f"Backend     : {cfg.backend}")
    print(f"Epoche      : {cfg.epochs}")
    print(f"{'='*60}\n")

    # ── 1. Carica il modello Keras salvato ───────────────────────────────────
    keras_model_path = experiment / "Model" / "best-model.keras"
    if not keras_model_path.exists():
        print(
            f"[AVVISO] {keras_model_path} non trovato. "
            "Salto la valutazione del modello pre-addestrato."
        )
        saved_model = None
    else:
        print(f"\n[1] Caricamento modello da: {keras_model_path}")
        from test_utils import load_keras_model
        saved_model = load_keras_model(str(keras_model_path))
        print("[1] Modello caricato.")

    # ── 2. Carica il dataset ─────────────────────────────────────────────────
    print(f"\n[2] Caricamento dataset '{cfg.dataset}'...")
    dataset = _load_dataset(cfg.dataset)
    dataset.data_as_float32()
    print(f"[2] Dataset caricato: {dataset.X_train.shape[0]} train / {dataset.X_test.shape[0]} test campioni.")

    # ── 3. Testa il modello salvato ──────────────────────────────────────────
    if saved_model is not None:
        print("\n[3] Valutazione del modello pre-addestrato (best-model.keras)...")
        saved_loss, saved_acc = _eval_keras_model(saved_model, dataset, cfg)
    else:
        print("\n[3] Nessun modello pre-addestrato da valutare.")

    # ── 4. Estrai il punto dell'iterazione migliore ──────────────────────────
    algo_logs = experiment / "algorithm_logs"
    print(f"\n[4] Ricerca iterazione migliore in {algo_logs} ...")
    best_idx = _find_best_iteration(algo_logs)
    best_params = _load_best_params(algo_logs, best_idx)
    layer_x_block = _find_layer_x_block(experiment, best_idx)
    print(f"[4] layer_x_block={layer_x_block}")

    # ── 5–6. Ricrea il modello con il backend scelto e ri-addestra ───────────
    print(f"\n[5] Ricreazione del modello con backend='{cfg.backend}'...")

    if cfg.backend == "tf":
        from tensorflow_implementation import module_backend, neural_network
        from tensorflow.keras import backend as K
        K.clear_session()
    elif cfg.backend == "torch":
        from pytorch_implementation import module_backend, neural_network
    else:
        print(f"[ERRORE] Backend non supportato: {cfg.backend}", file=sys.stderr)
        sys.exit(1)

    backend_instance = module_backend.ModuleBackend()
    nn = neural_network.NeuralNetwork(
        backend=backend_instance,
        dataset=dataset,
        da=False,
        reg=False,
        residual=False,
    )

    nn.build_network(best_params, layer_x_block=layer_x_block)
    print(f"[5] Modello costruito (backend={cfg.backend}).")

    print(f"\n[6] Ri-addestramento per {cfg.epochs} epoche...")
    score, history, trained_model = nn.training(best_params)

    retrain_loss = float(score[0])
    retrain_acc = float(score[1])
    print(f"\n[6] Risultati ri-addestramento: loss={retrain_loss:.4f}  accuracy={retrain_acc:.4f}")

    # ── Riepilogo ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RIEPILOGO")
    if saved_model is not None:
        print(f"  Modello pre-addestrato  →  loss={saved_loss:.4f}  acc={saved_acc:.4f}")
    print(f"  Modello ri-addestrato   →  loss={retrain_loss:.4f}  acc={retrain_acc:.4f}")
    print(f"  Iterazione usata        : {best_idx}")
    print(f"  layer_x_block           : {layer_x_block}")
    print(f"  Iperparametri           : {best_params}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
