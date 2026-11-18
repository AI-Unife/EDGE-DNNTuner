"""
Reproducibility check (Riccardo's idea): retrain the SAME architecture several times with the SAME
random seed and compare the accuracy before and after 3-bit quantization. Optionally also with
DIFFERENT seeds, to see how much the accuracy varies from training to training (this is the
variance the tuner has to live with when it compares quantizers).

The architecture is taken from one of Alice's .keras files (config only, weights re-initialized),
including the data augmentation of the tuner (RandomFlip + RandomTranslation). Training uses
Adam(lr=1e-3), categorical cross-entropy, batch 64, early stopping on the validation accuracy
(patience 5, restore best weights) -- NOT the tuner's exact hyper-parameters (unknown for these
models), so the absolute accuracy will differ from Alice's; what matters here is run-to-run equality.

NOTE: the tuner itself does not seed TensorFlow at all (cfg.seed is used only by the BO sampler),
so its training is not reproducible as is. What this script does to make a run reproducible:
    1. PYTHONHASHSEED=0 in the environment (set before Python starts, see below)
    2. tf.keras.utils.set_random_seed(seed)        -> python random, numpy, tf, keras layers
    3. tf.config.experimental.enable_op_determinism() -> deterministic TF ops (slower)
    4. shuffling with a fixed seed, augmentation layers with an explicit seed

Run from the project root (set the hash seed in the SAME command):
    PowerShell:  $env:PYTHONHASHSEED="0"; python experiments/fp_quantizers/run_seed_repro.py --models-dir DIR 3 --runs 2 --epochs 5
    cmd:         set PYTHONHASHSEED=0 && python experiments/fp_quantizers/run_seed_repro.py --models-dir DIR 3 --runs 2 --epochs 5
Options:
    <rank>            which of Alice's models gives the architecture (default 3)
    --runs N          trainings with the same seed (default 2)
    --seeds a,b,c     additionally one training per seed in this list (default: none)
    --epochs E        max epochs per training (default 5; use 30+ for converged networks)
    --seed S          the seed for the same-seed runs (default 42)
    --bits 3|8|all    which quantizers to evaluate (default 3)
    --no-cache        retrain even if the weights for (architecture, seed, epochs) are saved
Trained weights are cached in seed_repro_cache/ (training is deterministic for a given seed), so a
second call with the same seed/epochs, e.g. with --bits 8 after --bits 3, does not train again.
Results: printed table + seed_repro_results.csv (appended).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _common  # noqa: F401,E402  (project root on sys.path, --models-dir)
MODELS_DIR = _common.models_dir_from_argv()

import argparse
import csv
import glob
import importlib
import json
import os
import sys
import zipfile

import numpy as np
import tensorflow as tf

from quantizer.quantizer_selector import CANDIDATES, sqnr_empirical_db, weight_tensors

OUT_CSV = "seed_repro_results.csv"
NORMALIZE = False          # like the efficient_BO branch (raw 0-255 pixels)
BATCH = 64

# 3-bit quantizers: 3 formats x (std, pow2) + the 12 variance-range quantizers
QUANT_3 = [(f"quantizer.quantizer_{f}", m) for f in ("fp120", "fp111", "fp102") for m in ("std", "pow2")]
QUANT_3 += [(c, None) for c in CANDIDATES]
# 8-bit quantizers: wide formats without scaling (their SQNR does not depend on sigma),
# FP(1,4,3) with and without scaling, FP(1,0,7) with scaling
QUANT_8 = [(f"quantizer.quantizer_{f}", "none") for f in ("fp170", "fp161", "fp152")]
QUANT_8 += [("quantizer.quantizer_fp143", m) for m in ("std", "pow2", "none")]
QUANT_8 += [("quantizer.quantizer_fp107", m) for m in ("std", "pow2")]
CACHE_DIR = "seed_repro_cache"   # trained weights per (architecture, seed, epochs): training is deterministic

ap = argparse.ArgumentParser()
ap.add_argument("rank", nargs="?", type=int, default=3)
ap.add_argument("--runs", type=int, default=2)
ap.add_argument("--seeds", type=str, default="")
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--bits", choices=["3", "8", "all"], default="3", help="which quantizers to evaluate")
ap.add_argument("--no-cache", action="store_true", help="always retrain (ignore saved weights)")
args = ap.parse_args()

QUANT = {"3": QUANT_3, "8": QUANT_8, "all": QUANT_3 + QUANT_8}[args.bits]

if os.environ.get("PYTHONHASHSEED") != "0":
    print("[WARNING] PYTHONHASHSEED is not 0: set it in the same command (see the docstring). "
          "Keras itself is still seeded, but full determinism is not guaranteed.")


# ----------------------------------------------------------------------------- architecture
def build_fresh_model(path, seed, augmentation=True):
    """Same architecture as the .keras file, freshly initialized (no weights loaded)."""
    with zipfile.ZipFile(path) as z:
        config = json.loads(z.read("config.json"))
    if config["class_name"] != "Sequential":
        raise ValueError("this script supports the Sequential models (not the HW3 ones with skip connections)")
    layers_cfg = [l for l in config["config"]["layers"] if l["class_name"] not in ("Sequential", "InputLayer")]
    model = tf.keras.Sequential(name=f"repro_seed{seed}")
    model.add(tf.keras.layers.InputLayer(input_shape=(32, 32, 3)))
    if augmentation:   # the tuner's augmentation (tensorflow_implementation/model.py), with explicit seeds
        model.add(tf.keras.layers.RandomFlip("horizontal", seed=seed))
        model.add(tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="nearest", seed=seed))
    for l in layers_cfg:
        l["config"].pop("name", None)          # fresh names, avoids clashes with augmentation layers
        model.add(tf.keras.layers.deserialize(l))
    return model


def set_all_seeds(seed):
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as e:  # noqa: BLE001
        print(f"[WARNING] enable_op_determinism not available: {e}")


# ----------------------------------------------------------------------------- data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
div = 255.0 if NORMALIZE else 1.0
x_train, x_test = x_train.astype("float32") / div, x_test.astype("float32") / div
y_train = tf.keras.utils.to_categorical(y_train.reshape(-1), 10)
y_test = tf.keras.utils.to_categorical(y_test.reshape(-1), 10)

files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.keras")))
path = [f for f in files if int(os.path.basename(f)[:2]) == args.rank][0]
print(f"architecture from {os.path.basename(path)}")


# ----------------------------------------------------------------------------- one training
def _core_of(model):
    """The network without the augmentation layers (they have no weights)."""
    core = tf.keras.Sequential([l for l in model.layers if not isinstance(
        l, (tf.keras.layers.RandomFlip, tf.keras.layers.RandomTranslation))])
    core.build((None, 32, 32, 3))
    return core


def train_once(seed, epochs, tag):
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = f"arch{args.rank:02d}_seed{seed}_ep{epochs}_adam1e-3_b{BATCH}"
    wfile = os.path.join(CACHE_DIR, key + ".weights.h5")
    mfile = os.path.join(CACHE_DIR, key + ".json")
    if not args.no_cache and os.path.exists(wfile) and os.path.exists(mfile):
        print(f"\n=== {tag}: seed={seed}, epochs<={epochs}  -> loaded from cache {wfile}")
        set_all_seeds(seed)
        core = _core_of(build_fresh_model(path, seed, augmentation=False))
        core.load_weights(wfile)
        n_epochs = json.load(open(mfile))["epochs_run"]
    else:
        set_all_seeds(seed)
        model = build_fresh_model(path, seed)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
        ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
              .shuffle(len(x_train), seed=seed, reshuffle_each_iteration=True).batch(BATCH))
        es = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, mode="max")
        print(f"\n=== {tag}: seed={seed}, epochs<={epochs}")
        hist = model.fit(ds, epochs=epochs, validation_data=(x_test, y_test), callbacks=[es], verbose=2)
        n_epochs = len(hist.history["loss"])
        core = _core_of(model)
        core.save_weights(wfile)
        json.dump({"epochs_run": n_epochs, "seed": seed, "epochs_max": epochs, "rank": args.rank}, open(mfile, "w"))
        print(f"    weights saved to {wfile}")
    core.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    _, f_acc = core.evaluate(x_test, y_test, verbose=0)
    rows = {"float": f_acc}
    for modname, mode in QUANT:
        cls = importlib.import_module(modname).quantizer_module
        q = cls(opt="adam") if mode is None else cls(opt="adam", scale=mode)
        qm = q.quantizer_function(core)
        _, acc = qm.evaluate(x_test, y_test, verbose=0)
        rows[f"{modname.split('.')[-1]}|{q.scale_mode}"] = acc
    return f_acc, n_epochs, core.get_weights(), rows


results = []   # (tag, seed, float_acc, epochs_run, weights, rows)
for r in range(args.runs):
    results.append(("same-seed run %d" % (r + 1), args.seed) + train_once(args.seed, args.epochs, f"same-seed run {r + 1}"))
for s in [int(v) for v in args.seeds.split(",") if v.strip()]:
    results.append((f"seed {s}", s) + train_once(s, args.epochs, f"seed {s}"))

# ----------------------------------------------------------------------------- report
print("\n" + "=" * 100)
print("WEIGHTS: max |w_run1 - w_run_k| over all tensors (0 = bit-identical training)")
w0 = results[0][4]
for tag, seed, facc, ep, w, rows in results[1:]:
    print(f"  {tag:18s} max diff = {max(float(np.max(np.abs(a - b))) for a, b in zip(w0, w)):.3e}")

keys = list(results[0][5].keys())
print(f"\nACCURACY before / after quantization ({args.bits}-bit quantizers)")
print(f"  {'quantizer':30s}" + "".join(f"{t:>20s}" for t, *_ in results))
for k in keys:
    vals = [res[5][k] for res in results]
    spread = max(vals) - min(vals)
    print(f"  {k:30s}" + "".join(f"{v:20.4f}" for v in vals) +
          f"   spread {spread:.4f}   mean {np.mean(vals):.4f} +- {np.std(vals):.4f}")
print("  (epochs actually run: " + ", ".join(f"{t}: {ep}" for t, _, _, ep, *_ in results) + ")")

new = not os.path.exists(OUT_CSV)
with open(OUT_CSV, "a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["architecture", "run", "seed", "epochs_run", "quantizer", "acc", "bits_evaluated"])
    for tag, seed, facc, ep, _, rows in results:
        for k, v in rows.items():
            w.writerow([os.path.basename(path)[:2], tag, seed, ep, k, f"{v:.4f}", args.bits])
print(f"\nresults appended to {OUT_CSV}")
