"""
quantizer/distortion.py -- ALL formulas used to compare quantizers and to choose one.

This file is the single source of truth for: the quantizer grids (FP, integer, power-of-two, binary),
the exact distortion / SQNR for a Laplacian input, the scaling rules and the network-level SQNR
that quantizer_selector.py maximizes. Everything is NumPy only (no TensorFlow / PyTorch).

====================================================================================================
1. INPUT MODEL
====================================================================================================
The weights of a tensor are modelled as a zero-mean Laplacian with standard deviation sigma:

        f(x) = 1/(2b) * exp(-|x| / b),      b = sigma / sqrt(2)            (var = 2 b^2 = sigma^2)

sigma of a real tensor is its RMS:  sigma = sqrt(mean(w^2)).
Variance in dB:  sigma^2 [dB] = 10 log10(sigma^2) = 20 log10(sigma).

====================================================================================================
2. QUANTIZER MODEL (a "Grid")
====================================================================================================
Every quantizer here is symmetric, so it is described by its non-negative half:

    levels  y_0 < y_1 < ... < y_{N-1}      (y_j >= 0; with a zero level y_0 = 0 -> "mid-tread")
    edges   a_0 = 0 < a_1 < ... < a_{N-1} < a_N = +inf

Cell j is [a_j, a_{j+1}) and every |x| in it is reconstructed as y_j (x -> sign(x) * y_j). The last cell
extends to +inf (saturation). A value exactly on an edge goes to the LOWER cell. Nearest-level
rounding means a_j = (y_{j-1} + y_j) / 2.

Grid used in the tuner (unit grid; the real quantizer is  s * Q(x / s)  with a scale s, see 5.):
    fp_grid(e, m)     FP(1,e,m): levels 2^(E-bias) (1 + M/2^m), no zero, bias = 2^(e-1)-1 (0 if e = 0)
                      The three formats of the tuner are FP(1,2,0), FP(1,1,1), FP(1,0,2).
The formulas below hold for ANY symmetric scalar quantizer given as a Grid (with or without a zero
level), so another type (e.g. integer) can be added later as a new grid function (section 7).

====================================================================================================
3. EXACT DISTORTION FOR A LAPLACIAN INPUT  (D = E[(x - Q(x))^2])
====================================================================================================
By symmetry D = 2 * sum_j  integral_{a_j}^{a_{j+1}} (x - y_j)^2 f(x) dx  with f(x) = exp(-x/b)/(2b), and

    integral (x-y)^2 exp(-x/b) dx  =  -b exp(-x/b) [ (x-y)^2 + 2b (x-y) + 2 b^2 ]

so, with   T(a, y) = exp(-a/b) * [ (a-y)^2 + 2b (a-y) + 2 b^2 ]   and   T(inf, y) = 0 :

    D(sigma) = sum_{j=0}^{N-1} [ T(a_j, y_j) - T(a_{j+1}, y_j) ]                    (laplace_distortion)

For a grid WITHOUT a zero level and with nearest-level edges (the FP formats) this is identical to
the closed form
    D = sigma^2 - sqrt2 sigma y_1 + y_1^2 - sqrt2 sigma sum_k (y_{k+1} - y_k) exp(-sqrt2 t_k / sigma),
    t_k = (y_k + y_{k+1}) / 2,
(checked numerically in the self-test at the bottom of this file).

====================================================================================================
4. SQNR
====================================================================================================
    SQNR(sigma) = 10 log10( sigma^2 / D(sigma) )   [dB]                               (sqnr_db)

====================================================================================================
5. SCALING   (quantizer = s * Q(x / s), s > 0)
====================================================================================================
Quantizing x with the grid scaled by s is the same as quantizing x / s (std sigma / s) with the unit grid:

    D_s(sigma) = s^2 * D(sigma / s)          SQNR_s(sigma) = 10 log10( sigma^2 / D_s(sigma) )

Scaling rules (how s is chosen):
  "none"   s = 1
  "std"    per tensor, s = sigma_tensor / sigma*,  where sigma* = best_sigma(grid) is the sigma that
           maximizes SQNR of the unit grid. Every tensor then works at the optimum of the format.
  "pow2"   same, with s rounded to the nearest power of two (a bit shift in hardware)
  "range"  ONE fixed s for the whole network, designed for weight variances sigma^2 in [a, b] dB:
           s maximizes the mean SQNR over the range, uniform in dB (best_range_scale):

               mean SQNR(s) = 1/(b-a) * integral_{a}^{b} SQNR_s( 10^(v/20) ) dv         (trapezoid rule)

           (optionally restricted to powers of two)

====================================================================================================
6. NETWORK-LEVEL SQNR AND QUANTIZER SELECTION  (what quantizer_selector.py maximizes)
====================================================================================================
For a trained network take every weight tensor i that is quantized (ndim > 1) with sigma_i = RMS,
n_i elements and scale s_i (given by the quantizer's scaling rule). Then D_i = D_{s_i}(sigma_i) and

    weighting "layer":   SQNR_net = mean_i  10 log10( sigma_i^2 / D_i )                      [dB]
    weighting "params":  SQNR_net = 10 log10( sum_i n_i sigma_i^2 / sum_i n_i D_i )          [dB]

The quantizer with the largest SQNR_net is selected. Assumption of the method: a higher SQNR of the
weights gives a higher accuracy of the quantized network. The empirical SQNR uses the same formulas
with D_i = mean((w - Q(w))^2) measured on the real weights.

====================================================================================================
7. HOW TO ADD A NEW QUANTIZER TYPE (e.g. integer)
====================================================================================================
(a) If it is a symmetric scalar quantizer, give its grid: write a function returning Grid(levels, edges,
    name) like fp_grid (for an integer quantizer: levels 0, 1, ..., 2^(b-1)-1 and edges at the half
    integers, with a zero level). Distortion, SQNR, scaling and the selection formulas above then work
    for it without any other change.
(b) Write the quantizer class (like quantizer/fp_quantizer_base.py) and set the attributes that
    quantizer_selector.py reads:  name, grid, scale_mode, range_scale, target_sigma, variance_range.
(c) Add its module name to CANDIDATES in quantizer/quantizer_selector.py.
A quantizer that is NOT of this form (asymmetric, non-uniform with sign-dependent levels, vector
quantizers) needs its own distortion formula: add it here and use it in the selector.

Run the self-test:  python -m quantizer.distortion
"""
import math
from collections import namedtuple

import numpy as np

SQRT2 = math.sqrt(2.0)
INF = float("inf")

Grid = namedtuple("Grid", ["levels", "edges", "name"])   # non-negative half of a symmetric quantizer


# ----------------------------------------------------------------------------------------------
# Grids
# ----------------------------------------------------------------------------------------------
def midpoint_edges(levels):
    """edges [0, midpoints between consecutive levels..., inf] (nearest-level rounding)."""
    y = np.asarray(levels, dtype=np.float64)
    return np.concatenate([[0.0], (y[:-1] + y[1:]) / 2.0, [INF]])


def fp_levels(e: int, m: int) -> np.ndarray:
    """Positive representable values of FP(1,e,m), ascending (float64). All 2^e exponent codes are
    normal numbers (no zero / inf / NaN / subnormals); bias = 2^(e-1)-1 for e >= 1, 0 for e = 0."""
    K = 2 ** m
    if e >= 1:
        L, bias = 2 ** e, 2 ** (e - 1) - 1
    else:
        L, bias = 1, 0
    return np.array(sorted(2.0 ** (E - bias) * (1.0 + M / K) for E in range(L) for M in range(K)),
                    dtype=np.float64)


def fp_grid(e: int, m: int) -> Grid:
    lv = fp_levels(e, m)
    return Grid(lv, midpoint_edges(lv), f"FP(1,{e},{m})")


# ----------------------------------------------------------------------------------------------
# Quantization (to check the formulas / for the empirical SQNR)
# ----------------------------------------------------------------------------------------------
def quantize(x, grid: Grid, scale: float = 1.0):
    """s * Q(x / s) for the symmetric grid (ties go to the lower cell, sign(0) = +)."""
    x = np.asarray(x)
    a = np.abs(x).astype(np.float64) / scale
    idx = np.searchsorted(grid.edges[1:-1], a, side="left")
    q = grid.levels[idx]
    return (np.where(x < 0, -q, q) * scale).astype(x.dtype)


# ----------------------------------------------------------------------------------------------
# Distortion and SQNR for a Laplacian input
# ----------------------------------------------------------------------------------------------
def _tail(a, y, b):
    """T(a, y) = exp(-a/b) [ (a-y)^2 + 2b (a-y) + 2 b^2 ]"""
    return np.exp(-a / b) * ((a - y) ** 2 + 2.0 * b * (a - y) + 2.0 * b ** 2)


def laplace_distortion(grid: Grid, sigma, scale: float = 1.0):
    """Exact D = E[(x - s Q(x/s))^2] for a zero-mean Laplacian with std sigma (scalar or array).
    D_s(sigma) = s^2 * D(sigma / s)."""
    sig = np.asarray(sigma, dtype=np.float64) / scale
    b = sig / SQRT2
    D = np.zeros_like(sig)
    for y, a, c in zip(grid.levels, grid.edges[:-1], grid.edges[1:]):
        D = D + _tail(a, y, b)
        if np.isfinite(c):
            D = D - _tail(c, y, b)
    D = D * scale ** 2
    return float(D) if D.ndim == 0 else D


def sqnr_db(grid: Grid, sigma, scale: float = 1.0):
    """SQNR [dB] = 10 log10(sigma^2 / D_s(sigma))."""
    sig = np.asarray(sigma, dtype=np.float64)
    out = 10.0 * np.log10(sig ** 2 / np.maximum(laplace_distortion(grid, sig, scale), 1e-300))
    return float(out) if out.ndim == 0 else out


def mean_sqnr_range_db(grid: Grid, db_min: float, db_max: float, scale: float = 1.0, num: int = 2001) -> float:
    """Mean SQNR [dB] over sigma^2 in [db_min, db_max] dB, uniform weighting in dB (trapezoid rule)."""
    db = np.linspace(db_min, db_max, num)
    c = sqnr_db(grid, 10 ** (db / 20), scale)
    trapz = getattr(np, "trapezoid", None) or np.trapz     # NumPy 1.x: trapz, 2.x: trapezoid
    return float(trapz(c, db) / (db_max - db_min))


# ----------------------------------------------------------------------------------------------
# Scaling rules
# ----------------------------------------------------------------------------------------------
def best_sigma(grid: Grid, num: int = 2000) -> float:
    """sigma* that maximizes the SQNR of the unit grid (search on a log grid)."""
    pos = grid.levels[grid.levels > 0]
    sig = np.logspace(math.log10(1e-2 * pos.min()), math.log10(10.0 * grid.levels.max()), num)
    return float(sig[int(np.argmax(sqnr_db(grid, sig)))])


def best_range_scale(grid: Grid, db_min: float, db_max: float, pow2: bool = True):
    """Fixed scale s maximizing the mean SQNR over sigma^2 in [db_min, db_max] dB (section 5).
    Returns (s, mean_sqnr_db at s). pow2=True restricts s to powers of two."""
    if pow2:
        cands = [2.0 ** k for k in range(-24, 9)]
    else:
        cands = [10 ** (g / 20) for g in np.arange(-80.0, 30.0001, 0.25)]
    vals = [mean_sqnr_range_db(grid, db_min, db_max, c) for c in cands]
    i = int(np.argmax(vals))
    return float(cands[i]), float(vals[i])


def per_tensor_scale(sigma: float, target_sigma: float, pow2: bool = False) -> float:
    """s = sigma / target_sigma (target_sigma = best_sigma(grid) for the 'std'/'pow2' rules)."""
    if sigma == 0.0:
        return 1.0
    s = sigma / target_sigma
    return 2.0 ** round(math.log2(s)) if pow2 else s


# ----------------------------------------------------------------------------------------------
# Network-level SQNR (section 6)
# ----------------------------------------------------------------------------------------------
def aggregate_sqnr_db(sigma2, dist, counts, weighting: str = "layer") -> float:
    """sigma2, dist, counts: per-tensor arrays (variance, distortion, number of weights)."""
    sigma2, dist, counts = (np.asarray(v, dtype=np.float64) for v in (sigma2, dist, counts))
    if weighting == "params":
        return 10 * math.log10(float(np.sum(counts * sigma2)) / max(float(np.sum(counts * dist)), 1e-300))
    if weighting == "layer":
        return float(np.mean(10 * np.log10(sigma2 / np.maximum(dist, 1e-300))))
    raise ValueError("weighting must be 'layer' or 'params'")


def network_sqnr_theory_db(grid: Grid, sigmas, counts, scales, weighting: str = "layer") -> float:
    """Theoretical SQNR of a network: per-tensor sigma_i, n_i and scale s_i -> D_i = s_i^2 D(sigma_i / s_i)."""
    sigmas = np.asarray(sigmas, dtype=np.float64)
    dist = [laplace_distortion(grid, s, c) for s, c in zip(sigmas, scales)]
    return aggregate_sqnr_db(sigmas ** 2, dist, counts, weighting)


def network_sqnr_empirical_db(grid: Grid, tensors, scales, weighting: str = "layer") -> float:
    """Same with the distortion measured on the real weights: D_i = mean((w - s Q(w / s))^2)."""
    sig2, dist, cnt = [], [], []
    for w, s in zip(tensors, scales):
        w = np.asarray(w, dtype=np.float64)
        sig2.append(float(np.mean(w ** 2)))
        dist.append(float(np.mean((w - quantize(w, grid, s)) ** 2)))
        cnt.append(w.size)
    return aggregate_sqnr_db(sig2, dist, cnt, weighting)


# ----------------------------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------------------------
def _laplace_distortion_fp_closed_form(levels, sigma):
    """Closed form of the PyTorch reference (section 3), only for midrise grids without a zero level."""
    y = [float(v) for v in levels]
    D = sigma ** 2 - SQRT2 * sigma * y[0] + y[0] ** 2
    for k in range(len(y) - 1):
        t = (y[k] + y[k + 1]) / 2
        D -= SQRT2 * sigma * (y[k + 1] - y[k]) * math.exp(-SQRT2 * t / sigma)
    return D


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("1) general formula == closed form of the FP reference (midrise, no zero level)")
    for e, m in ((2, 0), (1, 1), (0, 2)):
        g = fp_grid(e, m)
        err = max(abs(laplace_distortion(g, s) - _laplace_distortion_fp_closed_form(g.levels, s)) / s ** 2
                  for s in (0.05, 0.3, 1.0, 3.0, 20.0))
        print(f"   {g.name}: max relative difference = {err:.1e}")
        assert err < 1e-9

    print("\n2) Monte Carlo check of the exact SQNR [dB] (Laplacian samples), with and without a scale")
    print(f"   {'grid':10s} {'scale':>6s} {'sigma':>6s} {'theory':>9s} {'MC':>9s}")
    for g, s in ((fp_grid(2, 0), 1.0), (fp_grid(1, 1), 0.25), (fp_grid(0, 2), 0.03125)):
        for sigma in (0.1, 0.5, 2.0):
            x = rng.laplace(0, sigma / SQRT2, 2_000_000)
            mse = np.mean((x - quantize(x, g, s)) ** 2)
            th, mc = sqnr_db(g, sigma, s), 10 * math.log10(sigma ** 2 / mse)
            print(f"   {g.name:10s} {s:6.2f} {sigma:6.2f} {th:8.3f}  {mc:8.3f}")
            assert abs(th - mc) < 0.06, (g.name, s, sigma, th, mc)

    print("\n3) best sigma* and maximum SQNR of every grid (per-tensor 'std' rule)")
    for g in (fp_grid(2, 0), fp_grid(1, 1), fp_grid(0, 2)):
        sg = best_sigma(g)
        print(f"   {g.name:10s} sigma* = {sg:7.3f}   SQNR max = {sqnr_db(g, sg):6.2f} dB")

    print("\n4) variance-range design (fixed power-of-two scale) for sigma^2 in [-20,20], [-30,30], [-35,5] dB")
    for g in (fp_grid(2, 0), fp_grid(1, 1), fp_grid(0, 2)):
        for a, b in ((-20, 20), (-30, 30), (-35, 5)):
            s, v = best_range_scale(g, a, b, pow2=True)
            print(f"   {g.name:10s} [{a:4d},{b:3d}] dB: s = 2^{round(math.log2(s)):3d}  mean SQNR = {v:5.2f} dB")
    print("\nOK")
