"""
Where are the weight variances of Alice's 30 CIFAR-10 models?  (no evaluation, no dataset: ~1 minute)

Run from the project root:   python experiments/fp_quantizers/survey_alice_weights.py --models-dir DIR
Prints sigma^2 [dB] of every weight tensor (ndim > 1) of every model, a summary per model, the overall
distribution, and the mean SQNR that FP(1,2,0) / FP(1,1,1) / FP(1,0,2) would get on this set of
variances with the best fixed power-of-two scale (i.e. a variance-range quantizer designed for the
range where these weights really are). Also writes alice_weight_variances.csv.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _common  # noqa: F401,E402  (project root on sys.path, --models-dir)
MODELS_DIR = _common.models_dir_from_argv()

import csv
import glob
import io
import json
import math
import os
import zipfile

import h5py
import numpy as np

from quantizer import distortion as dist

OUT_CSV = "alice_weight_variances.csv"


def weight_tensors_from_keras_file(path):
    """(layer name, array) for every weight tensor with ndim > 1, read directly from the .keras zip."""
    out = []
    with zipfile.ZipFile(path) as z:
        with h5py.File(io.BytesIO(z.read("model.weights.h5")), "r") as h:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset) and name.startswith("layers/") and obj.ndim > 1:
                    out.append((name.split("/")[1], obj[()].astype(np.float64)))
            h.visititems(visit)
    return out


files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.keras")))
all_db, rows = [], []
print(f"{'model':8s} {'#tensors':>8s} {'min':>7s} {'median':>7s} {'max':>7s}   sigma^2 [dB] per tensor")
for path in files:
    name = os.path.basename(path)[:2]
    tensors = weight_tensors_from_keras_file(path)
    dbs = [10 * math.log10(np.mean(w ** 2)) for _, w in tensors]
    all_db += dbs
    for (lname, w), db in zip(tensors, dbs):
        rows.append([name, lname, w.shape, w.size, f"{db:.2f}", f"{np.abs(w).max():.4f}"])
    print(f"{name:8s} {len(dbs):8d} {min(dbs):7.1f} {np.median(dbs):7.1f} {max(dbs):7.1f}   {[round(d, 1) for d in dbs]}")

with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "layer", "shape", "n_weights", "sigma2_db", "max_abs"])
    w.writerows(rows)

all_db = np.array(all_db)
print("\n=== all weight tensors of the 30 models ===")
print(f"n = {len(all_db)}   min {all_db.min():.1f}   5% {np.percentile(all_db, 5):.1f}   median {np.median(all_db):.1f}   "
      f"95% {np.percentile(all_db, 95):.1f}   max {all_db.max():.1f}  [dB]")
hist, edges = np.histogram(all_db, bins=np.arange(math.floor(all_db.min() / 5) * 5, math.ceil(all_db.max() / 5) * 5 + 5, 5))
print("histogram (5 dB bins):")
for h, lo, hi in zip(hist, edges[:-1], edges[1:]):
    print(f"  [{lo:5.0f}, {hi:4.0f}) {'#' * h} {h}")

print("\n=== variance-range design on the REAL range of these weights ===")
lo, hi = float(np.percentile(all_db, 5)), float(np.percentile(all_db, 95))
lo_r, hi_r = 5 * math.floor(lo / 5), 5 * math.ceil(hi / 5)
print(f"range covering 5%-95% of the tensors: [{lo_r}, {hi_r}] dB")
for e, m in ((2, 0), (1, 1), (0, 2)):
    g = dist.fp_grid(e, m)
    for a, b in ((lo_r, hi_r), (-20, 20), (-30, 30), (-35, 5)):
        s, v = dist.best_range_scale(g, a, b, pow2=True)
        # mean SQNR actually obtained on the real set of variances with this fixed scale
        real = float(np.mean(dist.sqnr_db(g, 10 ** (all_db / 20), s)))
        print(f"  {g.name}  designed for [{a:4.0f},{b:4.0f}] dB: c = 2^{round(math.log2(s)):3d}   "
              f"mean SQNR on design range {v:6.2f} dB   mean SQNR on the real weights {real:6.2f} dB")
print(f"\nwritten {OUT_CSV}")
