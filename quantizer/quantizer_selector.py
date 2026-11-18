"""
Selects, for a trained network, the quantizer with the highest SQNR.

The SQNR of a quantizer depends on the variance of the weights it quantizes. For each candidate the
SQNR of the network's weights is computed and the candidate with the highest value is chosen.
Assumption: higher SQNR -> higher accuracy of the quantized network.

ALL formulas (grids, exact Laplacian distortion, SQNR, scaling rules, network-level SQNR) live in
quantizer/distortion.py -- read its docstring (sections 3-6) for the math. This file only applies them
to the weights of a model:
    per weight tensor i (ndim > 1): sigma_i = RMS, n_i = size, s_i = scale of the candidate's scaling rule
    D_i = s_i^2 * D(sigma_i / s_i)            SQNR_net = aggregate over tensors ("layer" / "params")

method="theory"    selects with the SQNR computed from the variances (Laplacian assumption);
method="empirical" selects with the SQNR measured by quantizing the real weights.
Both values are always computed and printed.

A candidate is a module with a class `quantizer_module` that has (see FPQuantizerBase):
    name, grid (distortion.Grid), scale_mode ("none"|"std"|"pow2"|"range"), range_scale, target_sigma,
    variance_range, and a constructor quantizer_module(opt=..., quantize_bias=...).
To add another quantizer family (e.g. integer) see distortion.py, section 7.

Usage (see components/controller.py) -- the same calling convention as the other quantizers of the
tuner (`import quantizer.X as quantizer_module; quantizer_module.quantizer_module(opt=...)`):
    quantizer_module, report = select_best_quantizer(self.model)   # module with the highest SQNR
    quantizer = quantizer_module.quantizer_module(opt=params["optimizer"])
"""
import importlib
import math
import os

import numpy as np

from quantizer import distortion as dist
from quantizer.fp_quantizer_base import tensor_scale

# Quantizers that compete. Edit this list to add/remove candidates.
# Per-tensor-scaled quantizers (quantizer_fp120.py, ... with scale "std"/"pow2") could be added here too,
# but they rescale every tensor to the same optimal sigma, so the one with the best format would always win.
CANDIDATES = [
    f"quantizer.quantizer_{fmt}_{rng}"
    for fmt in ("fp120", "fp111", "fp102")
    for rng in ("m20_p20", "m30_p30", "m35_p5", "m50_m15")   # sigma^2 ranges [-20,20], [-30,30], [-35,5], [-50,-15] dB
]


def keras_model_of(model):
    """The tuner passes a wrapper (TFModel) whose Keras model is in `.model`. Returns the Keras
    model (an object with get_weights), or None (e.g. for the PyTorch backend)."""
    if hasattr(model, "get_weights"):
        return model
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "get_weights"):
        return inner
    return None


def _scale_for(q, sigma):
    """Scale s used by quantizer q for a tensor with RMS sigma (mirrors FPQuantizerBase.quantizer_function)."""
    if q.scale_mode == "none":
        return 1.0
    if q.scale_mode == "range":
        return q.range_scale
    return tensor_scale(np.full(1, sigma), q.target_sigma, q.scale_mode)


def weight_tensors(model, quantize_bias=False):
    """Weight tensors that the quantizers would quantize (ndim > 1, or all if quantize_bias)."""
    return [np.asarray(w, dtype=np.float64) for w in model.get_weights()
            if (w.ndim > 1 or quantize_bias) and w.size > 0]


def _tensors_and_scales(q, tensors):
    """Non-zero tensors with their RMS, size and the scale chosen by the quantizer's rule."""
    kept, sigmas, counts, scales = [], [], [], []
    for w in tensors:
        sigma = float(np.sqrt(np.mean(w ** 2)))
        if sigma == 0.0:
            continue
        kept.append(w)
        sigmas.append(sigma)
        counts.append(w.size)
        scales.append(_scale_for(q, sigma))
    return kept, sigmas, counts, scales


def sqnr_theory_db(q, tensors, weighting="layer"):
    kept, sigmas, counts, scales = _tensors_and_scales(q, tensors)
    if not kept:
        return float("nan")
    return dist.network_sqnr_theory_db(q.grid, sigmas, counts, scales, weighting)


def sqnr_empirical_db(q, tensors, weighting="layer"):
    kept, sigmas, counts, scales = _tensors_and_scales(q, tensors)
    if not kept:
        return float("nan")
    return dist.network_sqnr_empirical_db(q.grid, kept, scales, weighting)


def select_best_quantizer(model, candidates=None, method="theory", weighting="layer",
                          quantize_bias=False, verbose=True, log_dir=None):
    """
    Args:
        model: Keras model (or the tuner's wrapper, see keras_model_of).
        candidates: list of module names (default CANDIDATES).
        method: "theory" (SQNR from the variance of the weights) or "empirical" (measured on the weights).
        weighting: "layer" or "params" (see distortion.py, section 6).
        log_dir: if given, the selection is appended to <log_dir>/quantizer_selection.txt.
    Returns:
        (quantizer_module, report): quantizer_module is the imported MODULE with the highest SQNR
        (use quantizer_module.quantizer_module(opt=...) to build the quantizer, like the other
        quantizers of the tuner); report is a list of dicts sorted by the selection SQNR
        (descending), report[0] is the selected one.
    """
    if method not in ("theory", "empirical"):
        raise ValueError("method must be 'theory' or 'empirical'")
    if weighting not in ("layer", "params"):
        raise ValueError("weighting must be 'layer' or 'params'")
    keras_model = keras_model_of(model)
    if keras_model is None:
        raise ValueError("select_best_quantizer needs a Keras model (get_weights); got %r" % type(model))

    tensors = weight_tensors(keras_model, quantize_bias)
    if not tensors:
        raise ValueError("model has no weight tensors to quantize")

    report = []
    for name in (candidates or CANDIDATES):
        mod = importlib.import_module(name)
        cls = mod.quantizer_module
        q = cls(opt="adam", quantize_bias=quantize_bias)
        th = sqnr_theory_db(q, tensors, weighting)
        em = sqnr_empirical_db(q, tensors, weighting)
        report.append({"module": name, "name": q.name,
                       "variance_range": q.variance_range, "scale": q.range_scale if q.scale_mode == "range" else None,
                       "sqnr_theory_db": th, "sqnr_empirical_db": em, "module_obj": mod, "cls": cls})
    key = "sqnr_theory_db" if method == "theory" else "sqnr_empirical_db"
    report.sort(key=lambda r: -r[key] if not math.isnan(r[key]) else float("inf"))
    best = report[0]

    if verbose:
        sig_db = [round(10 * math.log10(float(np.mean(w ** 2))), 1) for w in tensors if np.any(w)]
        print(f"[QUANTIZER SELECTION] method={method}, weighting={weighting}; "
              f"weight sigma^2 per tensor [dB]: {sig_db}")
        print(f"  {'module':34s} {'SQNR theory':>12s} {'SQNR empirical':>15s}")
        for r in report:
            mark = "  <== selected" if r is best else ""
            print(f"  {r['module'].split('.')[-1]:34s} {r['sqnr_theory_db']:10.2f} dB {r['sqnr_empirical_db']:12.2f} dB{mark}")
        if best[key] <= 0:
            print("  [WARNING] even the best candidate has SQNR <= 0 dB: the weight variances are far from "
                  "the variance ranges the candidates were designed for.")

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "quantizer_selection.txt"), "a") as f:
            f.write(f"{best['module'].split('.')[-1]},{best['sqnr_theory_db']:.4f},{best['sqnr_empirical_db']:.4f}\n")
    return best["module_obj"], report


if __name__ == "__main__":
    ### Demo without TensorFlow: a fake model whose weight tensors have the given sigma^2 [dB].
    ### Run from the project root:  python -m quantizer.quantizer_selector
    class _FakeModel:
        def __init__(self, sigma2_db, seed=0):
            rng = np.random.default_rng(seed)
            self._w = [rng.laplace(0, 10 ** (db / 20) / math.sqrt(2), size=(64, 64)).astype("float32")
                       for db in sigma2_db]
            self._w.append(np.zeros(64, dtype="float32"))   # a bias (ignored)

        def get_weights(self):
            return self._w

    for dbs in ([-24, -38, -39, -42, -43, -34], [-10, -5, 0, 3], [-20, -10, 0, 10, 15]):
        print(f"\n=== network with weight sigma^2 [dB] = {dbs}")
        quantizer_module, rep = select_best_quantizer(_FakeModel(dbs))
        print("selected:", quantizer_module.__name__, "->", quantizer_module.quantizer_module)
