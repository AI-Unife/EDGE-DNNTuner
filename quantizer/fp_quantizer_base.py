"""
Shared code for the 3-bit floating-point quantizers FP(1, e, m) (no scaling).

NumPy port of the PyTorch reference (fp3_quantizers.py). Used by
    quantizer_fp120.py  -> FP(1,2,0)
    quantizer_fp111.py  -> FP(1,1,1)
    quantizer_fp102.py  -> FP(1,0,2)

Convention (same as the PyTorch reference):
  * all 2^e exponent codes are normal numbers (no zero, inf, NaN, subnormals)
  * bias = 2^(e-1) - 1 for e >= 1, bias = 0 for e = 0
  * positive levels: v = 2^(E - bias) * (1 + M / 2^m)
  * rounding to nearest level (thresholds = midpoints), saturation at x_max
  * x -> sign(x) * Q(|x|), with sign(0) taken as +
  * NO scaling of the weights
"""
import math
import os

import numpy as np

from quantizer import distortion as dist
from quantizer.quantizer_interface import quantizer_interface
from exp_config import load_cfg


# ----------------------------------------------------------------------------
# Numeric core: thin wrappers around quantizer/distortion.py (single source of the formulas)
# ----------------------------------------------------------------------------
fp_levels = dist.fp_levels


def _grid(levels) -> dist.Grid:
    levels = np.asarray(levels, dtype=np.float64)
    return dist.Grid(levels, dist.midpoint_edges(levels), "FP")


def fp_thresholds(levels: np.ndarray) -> np.ndarray:
    """Decision thresholds = midpoints between consecutive levels."""
    return dist.midpoint_edges(levels)[1:-1]


def fp_quantize(x, levels: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Quantize an array onto the FP grid. Same behaviour as torch.bucketize in the
    PyTorch reference: a value exactly on a threshold goes to the lower level."""
    x = np.asarray(x)
    a = np.abs(x).astype(np.float64)
    idx = np.searchsorted(thresholds, a, side="left")  # 0..N-1, last cell open to +inf
    q = levels[idx]
    q = np.where(x < 0, -q, q)
    return q.astype(x.dtype)


def laplace_distortion(levels, sigma: float) -> float:
    """Exact D(sigma) for a zero-mean Laplacian with std sigma (see distortion.py, section 3)."""
    return dist.laplace_distortion(_grid(levels), sigma)


def laplace_sqnr_db(levels, sigma: float) -> float:
    return dist.sqnr_db(_grid(levels), sigma)


def best_sigma(levels) -> float:
    """Std of a zero-mean Laplacian that maximizes the exact SQNR of the format."""
    return dist.best_sigma(_grid(levels))


def laplace_sqnr_curve(levels, sigmas) -> np.ndarray:
    """Vectorized exact SQNR [dB] for every sigma in `sigmas`."""
    return dist.sqnr_db(_grid(levels), sigmas)


def mean_sqnr_db(levels, db_min: float, db_max: float, num: int = 2001) -> float:
    """Mean SQNR [dB] over sigma^2 in [db_min, db_max] dB, uniform weighting in dB."""
    return dist.mean_sqnr_range_db(_grid(levels), db_min, db_max, 1.0, num)


def best_range_scale(levels, db_min: float, db_max: float, pow2: bool = True):
    """Fixed grid scale c maximizing the mean SQNR over sigma^2 in [db_min, db_max] dB.
    Returns (c, mean_sqnr_db_at_c)."""
    return dist.best_range_scale(_grid(levels), db_min, db_max, pow2)


def tensor_scale(w, target_sigma: float, mode: str = "std") -> float:
    """Per-tensor scale s such that the std (RMS, zero mean assumed) of w / s equals target_sigma.
    mode 'std': s as computed; mode 'pow2': s rounded to the nearest power of two."""
    rms = float(np.sqrt(np.mean(np.square(np.asarray(w, dtype=np.float64)))))
    return dist.per_tensor_scale(rms, target_sigma, pow2=(mode == "pow2"))


# ----------------------------------------------------------------------------
# Tuner quantizer (Keras model in, quantized Keras model out)
# ----------------------------------------------------------------------------
class FPQuantizerBase(quantizer_interface):
    """Base class. Subclasses only set E and M (see quantizer_fp120.py etc.)."""

    E = None
    M = None
    SCALE = "std"   # default scaling mode: "none" | "std" | "pow2" | "range"
    VARIANCE_RANGE = None   # (db_min, db_max) of sigma^2 in dB -> variance-range quantizer (see below)
    RANGE_POW2 = True       # variance-range quantizer: restrict the fixed scale c to a power of two

    def __init__(self, opt="adam", quantize_bias=False, scale=None, target_sigma=None,
                 variance_range=None, range_pow2=None):
        """
        Args:
            opt: optimizer used to compile the quantized model (name given by the tuner).
            quantize_bias: if True also 1-D tensors (biases, BatchNorm params) are quantized.
                           Default False, like quantizer_POTQ.py (only tensors with ndim > 1).
            scale: per-tensor scaling of the weights before quantization, w ~ s * Q(w / s):
                   "none" -> no scaling (s = 1)
                   "std"  -> s so that std(w / s) = target_sigma
                   "pow2" -> same, with s rounded to a power of two (bit shift on FPGA)
                   None   -> class default SCALE.
            target_sigma: sigma used by "std"/"pow2". Default: the sigma that maximizes the
                          exact Laplacian SQNR of the format (best_sigma).
            variance_range: (db_min, db_max), the range of sigma^2 [dB] the quantizer is designed
                          for. Selects scale="range": ONE fixed scale c for all tensors (no
                          per-tensor statistics), chosen to maximize the mean SQNR over the range.
            range_pow2: restrict c to a power of two (default: class RANGE_POW2 = True).
        """
        if self.E is None or self.M is None:
            raise NotImplementedError("Subclass must define E and M")
        self.opt = opt
        self.quantize_bias = quantize_bias
        self.variance_range = variance_range if variance_range is not None else self.VARIANCE_RANGE
        self.range_pow2 = self.RANGE_POW2 if range_pow2 is None else range_pow2
        if scale is None:
            scale = "range" if self.variance_range is not None else self.SCALE
        self.scale_mode = scale
        if self.scale_mode not in ("none", "std", "pow2", "range"):
            raise ValueError("scale must be 'none', 'std', 'pow2' or 'range'")
        if self.scale_mode == "range" and self.variance_range is None:
            raise ValueError("scale='range' needs variance_range=(db_min, db_max)")
        self.scales = []   # one scale per quantized tensor (needed later for the FPGA side)
        self.quantized_model = None
        self.score = None
        self.levels = fp_levels(self.E, self.M)
        self.thresholds = fp_thresholds(self.levels)
        self.grid = dist.Grid(self.levels, dist.midpoint_edges(self.levels), self.name)   # for distortion.py
        self.target_sigma = float(target_sigma) if target_sigma else best_sigma(self.levels)
        self.range_scale, self.range_mean_sqnr = None, None
        if self.scale_mode == "range":
            a, b = self.variance_range
            self.range_scale, self.range_mean_sqnr = best_range_scale(self.levels, a, b, self.range_pow2)
        # The tuner always has an active config; standalone tests may not.
        try:
            self.cfg = load_cfg()
        except FileNotFoundError:
            self.cfg = None
        self.level_usage = None   # fraction of quantized weights on each level (diagnostics)
        self.weight_mse = None

    @property
    def name(self):
        return f"FP(1,{self.E},{self.M})"

    @property
    def x_max(self):
        return float(self.levels[-1])

    def quantizer_function(self, model):
        import tensorflow as tf

        # The tuner may pass its model wrapper (TFModel): the Keras model is in `.model`
        if not hasattr(model, "get_weights") and hasattr(getattr(model, "model", None), "get_weights"):
            model = model.model
        weights = model.get_weights()
        new_weights = []
        counts = np.zeros(len(self.levels), dtype=np.int64)
        sq_err, n_tot = 0.0, 0

        self.scales = []
        for w in weights:
            if w.ndim > 1 or self.quantize_bias:
                w64 = w.astype(np.float64)
                if self.scale_mode == "none":
                    s = 1.0
                elif self.scale_mode == "range":
                    s = self.range_scale          # fixed for all tensors
                else:
                    s = tensor_scale(w64, self.target_sigma, self.scale_mode)
                q_unit = fp_quantize(w64 / s, self.levels, self.thresholds)   # values on the FP grid
                wq = (q_unit * s).astype(w.dtype)                              # back to the original range
                new_weights.append(wq)
                self.scales.append(s)
                idx = np.searchsorted(self.levels, np.abs(q_unit).ravel() - 1e-12)
                counts += np.bincount(np.clip(idx, 0, len(self.levels) - 1), minlength=len(self.levels))
                sq_err += float(np.sum((w64 - wq.astype(np.float64)) ** 2))
                n_tot += w.size
            else:
                new_weights.append(w)   # kept in float

        if n_tot:
            self.level_usage = counts / n_tot
            self.weight_mse = sq_err / n_tot

        self.quantized_model = tf.keras.models.clone_model(model)
        # Use the loss of the original model (the examples hard-code binary_crossentropy)
        loss = getattr(model, "loss", None) or "binary_crossentropy"
        self.quantized_model.compile(optimizer=self.opt, loss=loss, metrics=["accuracy"])
        self.quantized_model.set_weights(new_weights)
        return self.quantized_model

    def evaluate_quantized_model(self, x_test, y_test):
        if self.quantized_model is None:
            raise ValueError("Quantized model is not available. Please run quantizer_function first.")

        # Gesture dataset needs the custom evaluation (same as the other quantizers)
        eval_model = None
        if (self.cfg is not None and self.cfg.mode in ("fwdPass", "hybrid")
                and "gesture" in str(self.cfg.dataset)):
            try:
                from components.custom_train import eval_model
            except ImportError:
                eval_model = None   # not available in every branch of the tuner
        if eval_model is not None:
            self.score = eval_model(self.quantized_model, x_test, y_test)
        else:
            self.score = self.quantized_model.evaluate(x_test, y_test, verbose=2)
        return self.score

    def save_quantized_model(self, path=None):
        if self.quantized_model is None:
            return
        path = path or f"quantized_model_fp{self.E}{self.M}.keras"
        self.quantized_model.save(path)
        print(f"Model saved to {path}")

    def printing_values(self):
        if len(self.levels) <= 8:
            print(f"{self.name}: levels={self.levels.tolist()}  "
                  f"thresholds={self.thresholds.tolist()}  x_max={self.x_max}")
        else:
            print(f"{self.name}: {len(self.levels)} levels in [{self.levels[0]:.4g}, {self.levels[-1]:.4g}] "
                  f"(first: {[float(f'{v:.4g}') for v in self.levels[:4]]} ...)  x_max={self.x_max:.4g}")
        print(f"  scale mode: {self.scale_mode}   target sigma: {self.target_sigma:.3f}")
        if self.scale_mode == "range":
            a, b = self.variance_range
            print(f"  variance range sigma^2 in [{a}, {b}] dB: fixed c = {self.range_scale:.6g} "
                  f"(= 2^{math.log2(self.range_scale):.2f}), design mean SQNR = {self.range_mean_sqnr:.2f} dB")
        if self.scales and self.scale_mode != "range":
            if self.scale_mode == "pow2":
                print(f"  per-tensor scales as 2^k, k = {[int(round(math.log2(s))) for s in self.scales]}")
            else:
                print(f"  per-tensor scales: {[float(f'{s:.4g}') for s in self.scales]}")
        if self.level_usage is not None:
            if len(self.levels) <= 8:
                usage = {f"{l:g}": f"{u * 100:.1f}%" for l, u in zip(self.levels, self.level_usage)}
                print(f"  share of |weights| on each level: {usage}")
            else:
                used = int(np.sum(self.level_usage > 0))
                print(f"  levels actually used: {used} of {len(self.levels)}; share on lowest level "
                      f"{self.level_usage[0] * 100:.1f}%, on highest level {self.level_usage[-1] * 100:.1f}%")
            print(f"  weight MSE (float vs quantized): {self.weight_mse:.6g}")

    def plotting_function(self):
        pass

    def log_function(self):
        # Same format as quantizer_POTQ.py: one "loss,accuracy" line per evaluated network
        if self.cfg is None or self.score is None:
            return
        log_dir = "{}/algorithm_logs".format(self.cfg.name)
        os.makedirs(log_dir, exist_ok=True)
        with open("{}/quantization_report.txt".format(log_dir), "a") as f:
            f.write(str(self.score[0]) + "," + str(self.score[1]) + "\n")


# ----------------------------------------------------------------------------
# Standalone self-test shared by the three quantizer files
# ----------------------------------------------------------------------------
def run_selftest(quantizer_cls):
    """Run with: python -m quantizer.quantizer_fp120   (or Run in PyCharm, working dir = project root)."""
    q = quantizer_cls(opt="adam")
    print("=" * 70)
    q.printing_values()

    # 1) grid / ties / sign / saturation
    t = np.array([-100.0, -3.1, -0.1, 0.0, 0.1, 0.74, 0.76, 1.4, 1.6, 2.9, 3.1, 100.0], dtype=np.float32)
    print("x  :", t.tolist())
    print("Q(x):", fp_quantize(t, q.levels, q.thresholds).tolist())

    # 2) empirical SQNR on Laplacian data vs exact theory
    rng = np.random.default_rng(0)
    for s in (0.5, 1.0, 2.0):
        x = rng.laplace(0.0, s / math.sqrt(2), 2_000_000)
        mse = np.mean((x - fp_quantize(x, q.levels, q.thresholds)) ** 2)
        print(f"sigma={s:3}: SQNR empirical={10 * math.log10(s * s / mse):7.3f} dB   "
              f"theory={laplace_sqnr_db(q.levels, s):7.3f} dB")

    # 3) Keras model on MNIST 0/1 (same test as binary_quantizer.py)
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    k_tr, k_te = (y_train == 0) | (y_train == 1), (y_test == 0) | (y_test == 1)
    x_train, y_train, x_test, y_test = x_train[k_tr], y_train[k_tr], x_test[k_te], y_test[k_te]

    model = Sequential([Flatten(input_shape=(28, 28)), Dense(8, activation="relu"), Dense(1, activation="sigmoid")])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=5, verbose=0)
    loss_f, acc_f = model.evaluate(x_test, y_test, verbose=0)
    print(f"Float model      -> loss {loss_f:.4f}  acc {acc_f:.4f}")

    modes = [None, "none", "std", "pow2"]   # None = the quantizer's own default (e.g. its variance range)
    for mode in modes:
        q = quantizer_cls(opt="adam", scale=mode)
        q.quantizer_function(model)
        loss_q, acc_q = q.evaluate_quantized_model(x_test, y_test)
        print(f"{q.name} scale={q.scale_mode:5s}{' (default)' if mode is None else ''} -> "
              f"loss {loss_q:.4f}  acc {acc_q:.4f}")
        q.printing_values()
