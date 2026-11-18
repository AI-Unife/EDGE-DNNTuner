"""
Two experiments on Alice's pre-trained CIFAR-10 models (no training):

  A) BatchNorm re-calibration: after quantizing the weights, run ~N_CALIB training images through the
     network in training mode so that the BatchNorm layers refresh their moving mean / variance
     (weights are NOT changed). Only meaningful for models with BatchNorm (e.g. 26, 15).
  B) target_sigma sweep for FP(1,2,0) with per-tensor scaling ("std" and "pow2"): a smaller target
     sigma means a larger scale s = sigma_tensor / target_sigma, i.e. less saturation of the large
     weights. Checks whether the sigma that maximizes the SQNR of the weights (1.42) is also the one
     that maximizes the accuracy.

Run from the project root:
    python experiments/fp_quantizers/run_alice_experiments.py --models-dir DIR 26 01
    python experiments/fp_quantizers/run_alice_experiments.py --models-dir DIR 26 01 --no-bn      (skip experiment A)
Results are appended to alice_experiments.csv.
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

from alice_loader import load_alice_model
from quantizer.quantizer_selector import CANDIDATES, sqnr_empirical_db, weight_tensors

OUT_CSV = "alice_experiments.csv"
N_CALIB = 2000                      # training images used for the BN re-calibration
SIGMAS = [0.35, 0.5, 0.7, 1.0, 1.42, 2.0]   # target_sigma values for experiment B
NORMALIZE = False                   # Alice's models: raw 0-255 pixels

args = [a for a in sys.argv[1:] if not a.startswith("--")]
do_bn = "--no-bn" not in sys.argv

# ------------------------------------------------------------------ data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
div = 255.0 if NORMALIZE else 1.0
x_train, x_test = x_train.astype("float32") / div, x_test.astype("float32") / div
y_test = tf.keras.utils.to_categorical(y_test.reshape(-1), 10)
rng = np.random.default_rng(0)
x_calib = x_train[rng.choice(len(x_train), min(N_CALIB, len(x_train)), replace=False)]


def has_batchnorm(model):
    return any(isinstance(l, tf.keras.layers.BatchNormalization) for l in model.layers)


def recalibrate_bn(model, x, batch=100):
    """Refresh BN moving statistics with forward passes in training mode. Weights unchanged.
    Momentum is set to 0 during the passes so that the statistics are those of the calibration data."""
    bns = [l for l in model.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    old = [l.momentum for l in bns]
    for l in bns:
        l.momentum = 0.0
    for i in range(0, len(x), batch):
        model(x[i:i + batch], training=True)        # Dropout also active, harmless for statistics
    for l, m in zip(bns, old):
        l.momentum = m


def evaluate(model):
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    return float(acc)


files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.keras")))
chosen = [f for f in files if int(os.path.basename(f)[:2]) in {int(a) for a in args}] if args else files[:1]

new_csv = not os.path.exists(OUT_CSV)
with open(OUT_CSV, "a", newline="") as fcsv:
    wr = csv.writer(fcsv)
    if new_csv:
        wr.writerow(["model", "experiment", "quantizer", "scale", "target_sigma", "bn_recalib", "sqnr_emp_db", "acc"])

    for path in chosen:
        name = os.path.basename(path)
        print("\n" + "=" * 100 + f"\nMODEL {name}")
        model = load_alice_model(path)
        f_acc = evaluate(model)
        tensors = weight_tensors(model)
        bn = has_batchnorm(model)
        print(f"float accuracy: {f_acc:.4f}   BatchNorm: {bn}   weight sigma^2 [dB]: "
              f"{[round(10 * math.log10(np.mean(w ** 2)), 1) for w in tensors]}")

        # ---------------- A) BN re-calibration
        if do_bn and bn:
            print(f"\n--- A) BatchNorm re-calibration ({N_CALIB} training images), accuracy before -> after")
            configs = [("quantizer.quantizer_fp120", "std"), ("quantizer.quantizer_fp120", "pow2"),
                       ("quantizer.quantizer_fp111", "std")] + [(c, None) for c in CANDIDATES if "fp120" in c]
            for modname, mode in configs:
                cls = importlib.import_module(modname).quantizer_module
                q = cls(opt="adam") if mode is None else cls(opt="adam", scale=mode)
                qm = q.quantizer_function(model)
                acc0 = evaluate(qm)
                recalibrate_bn(qm, x_calib)
                acc1 = evaluate(qm)
                label = modname.split(".")[-1]
                print(f"  {label:26s} {q.scale_mode:6s} {acc0:.4f} -> {acc1:.4f}   ({acc1 - acc0:+.4f})")
                wr.writerow([name, "A_bn", label, q.scale_mode, "", 0, f"{sqnr_empirical_db(q, tensors):.3f}", f"{acc0:.4f}"])
                wr.writerow([name, "A_bn", label, q.scale_mode, "", 1, f"{sqnr_empirical_db(q, tensors):.3f}", f"{acc1:.4f}"])
                fcsv.flush()
        elif do_bn:
            print("\n--- A) skipped: no BatchNorm in this model")

        # ---------------- B) target_sigma sweep
        print(f"\n--- B) target_sigma sweep, FP(1,2,0) per-tensor scaling (SQNR-optimal sigma* = 1.42)")
        print(f"  {'target_sigma':12s} {'scale':6s} {'SQNR emp':>9s} {'acc':>7s}" + ("  acc after BN recalib" if (do_bn and bn) else ""))
        cls = importlib.import_module("quantizer.quantizer_fp120").quantizer_module
        for mode in ("std", "pow2"):
            for ts in SIGMAS:
                q = cls(opt="adam", scale=mode, target_sigma=ts)
                qm = q.quantizer_function(model)
                em = sqnr_empirical_db(q, tensors)
                acc = evaluate(qm)
                line = f"  {ts:12.2f} {mode:6s} {em:8.2f} {acc:8.4f}"
                acc_bn = ""
                if do_bn and bn:
                    recalibrate_bn(qm, x_calib)
                    acc_bn = evaluate(qm)
                    line += f"  {acc_bn:8.4f}"
                print(line)
                wr.writerow([name, "B_sigma", "quantizer_fp120", mode, ts, 0, f"{em:.3f}", f"{acc:.4f}"])
                if acc_bn != "":
                    wr.writerow([name, "B_sigma", "quantizer_fp120", mode, ts, 1, f"{em:.3f}", f"{acc_bn:.4f}"])
                fcsv.flush()
        print(f"\n  (float accuracy {f_acc:.4f})")

print(f"\nresults appended to {OUT_CSV}")
