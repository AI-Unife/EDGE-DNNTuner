"""
Test ALL FP quantizers on the pre-trained CIFAR-10 models from Alice (no training needed).

Run from the project root, with the Python that has TF 2.15 (Keras 2). DIR is the folder with the
.keras files (instead of --models-dir DIR you can set the environment variable ALICE_MODELS_DIR):
    python experiments/fp_quantizers/run_alice_test.py --models-dir DIR          # first model
    python experiments/fp_quantizers/run_alice_test.py --models-dir DIR 3        # model with rank 3
    python experiments/fp_quantizers/run_alice_test.py --models-dir DIR 1 3 12   # several models
    python experiments/fp_quantizers/run_alice_test.py --models-dir DIR all      # all 30 (slow)

For every model: float accuracy, sigma^2 [dB] of every weight tensor, then for every quantizer the
SQNR (theory / empirical) and the accuracy of the quantized model, plus the quantizer chosen by the
selector (highest SQNR). Results are appended to alice_quantizer_results.csv.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _common  # noqa: F401,E402  (project root on sys.path, --models-dir)
MODELS_DIR = _common.models_dir_from_argv()

import csv
import glob
import importlib
import math
import os
import sys

import numpy as np
import tensorflow as tf

from quantizer.quantizer_selector import select_best_quantizer, sqnr_theory_db, sqnr_empirical_db, weight_tensors, CANDIDATES
from alice_loader import load_alice_model   # the .keras files need a custom loader (nested Sequential)

OUT_CSV = "alice_quantizer_results.csv"

# ---------------------------------------------------------------- data
(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# Alice's models were trained with the efficient_BO branch, which feeds RAW pixels (0-255, no /255).
# Dividing by 255 gives ~10% accuracy. Set NORMALIZE=True for models trained with the `quantizer` branch.
NORMALIZE = False
x_test = x_test.astype("float32") / (255.0 if NORMALIZE else 1.0)
y_test = tf.keras.utils.to_categorical(y_test.reshape(-1), 10)

# ---------------------------------------------------------------- models
files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.keras")))
args = sys.argv[1:] or ["1"]
if args == ["all"]:
    chosen = files
else:
    chosen = [f for f in files if int(os.path.basename(f)[:2]) in {int(a) for a in args}]

# quantizers to compare: 3 formats x (none, std, pow2) + 9 variance-range quantizers
FORMATS = ["fp120", "fp111", "fp102",                    # 3-bit
           "fp170", "fp161", "fp152", "fp143", "fp107"]   # 8-bit
# scaling modes per format: the wide 8-bit formats do not need scaling (their SQNR does not depend on sigma)
SCALE_MODES = {"fp170": ("none",), "fp161": ("none",), "fp152": ("none",)}
CONFIGS = [(f"quantizer.quantizer_{fmt}", mode) for fmt in FORMATS for mode in SCALE_MODES.get(fmt, ("none", "std", "pow2"))]
CONFIGS += [(name, None) for name in CANDIDATES]

new_csv = not os.path.exists(OUT_CSV)
with open(OUT_CSV, "a", newline="") as fcsv:
    wr = csv.writer(fcsv)
    if new_csv:
        wr.writerow(["model", "float_acc", "quantizer", "scale", "sqnr_theory_db", "sqnr_empirical_db", "q_acc", "selected"])

    for path in chosen:
        name = os.path.basename(path)
        print("\n" + "=" * 100 + f"\nMODEL {name}")
        model = load_alice_model(path)
        f_loss, f_acc = model.evaluate(x_test, y_test, verbose=0)
        tensors = weight_tensors(model)
        print(f"float accuracy: {f_acc:.4f}   weight sigma^2 [dB] per tensor: "
              f"{[round(10 * math.log10(np.mean(w ** 2)), 1) for w in tensors]}")

        selected_module, _ = select_best_quantizer(model, verbose=False)
        selected = selected_module.__name__

        rows = []
        for modname, mode in CONFIGS:
            cls = importlib.import_module(modname).quantizer_module
            q = cls(opt="adam") if mode is None else cls(opt="adam", scale=mode)
            th, em = sqnr_theory_db(q, tensors), sqnr_empirical_db(q, tensors)
            q.quantizer_function(model)
            q_loss, q_acc = q.evaluate_quantized_model(x_test, y_test)
            label = modname.split(".")[-1]
            rows.append((label, q.scale_mode, th, em, q_acc, modname == selected and mode is None))
            wr.writerow([name, f"{f_acc:.4f}", label, q.scale_mode, f"{th:.3f}", f"{em:.3f}", f"{q_acc:.4f}", int(rows[-1][-1])])
            fcsv.flush()

        rows.sort(key=lambda r: -r[2])
        print(f"\n{'quantizer':26s} {'scale':6s} {'SQNR th':>8s} {'SQNR emp':>9s} {'acc':>7s}   (float acc {f_acc:.4f})")
        for label, mode, th, em, acc, sel in rows:
            print(f"{label:26s} {mode:6s} {th:7.2f} {em:8.2f} {acc:8.4f}{'   <== selected by SQNR' if sel else ''}")
        # does the highest SQNR give the highest accuracy? (among the 9 variance-range candidates)
        rng_rows = [r for r in rows if r[1] == "range"]
        best_by_sqnr = max(rng_rows, key=lambda r: r[2]); best_by_acc = max(rng_rows, key=lambda r: r[4])
        print(f"\nvariance-range candidates: best by SQNR = {best_by_sqnr[0]} (acc {best_by_sqnr[4]:.4f}); "
              f"best by accuracy = {best_by_acc[0]} (acc {best_by_acc[4]:.4f})")

print(f"\nresults appended to {OUT_CSV}")
