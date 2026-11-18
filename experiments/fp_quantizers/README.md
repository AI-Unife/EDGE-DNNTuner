# FP quantizers for the tuner: code and experiments

Floating-point weight quantizers FP(sign, exponent, mantissa) for the Symbolic DNN-Tuner, the
formulas used to compare them (exact SQNR for a Laplacian input), an SQNR-based selector, and the
scripts used to evaluate them on pre-trained CIFAR-10 networks.

All quantizers follow `quantizer/quantizer_interface.py`: the class is `quantizer_module`, so a
quantizer is switched by changing one import, e.g.
`from quantizer.quantizer_fp120 import quantizer_module`.

## Quantizers (`quantizer/`)

| file | content |
|---|---|
| `distortion.py` | **all the formulas**: FP grids, exact distortion and SQNR for a Laplacian input, scaling rules, network-level SQNR, how to add a new quantizer type. Self-test: `python -m quantizer.distortion` |
| `fp_quantizer_base.py` | common base class (Keras model in, quantized Keras model out); only weight tensors with `ndim > 1` are quantized, biases stay in float |
| `quantizer_fp120.py`, `quantizer_fp111.py`, `quantizer_fp102.py` | 3-bit FP(1,2,0), FP(1,1,1), FP(1,0,2) |
| `quantizer_fp170.py`, `quantizer_fp161.py`, `quantizer_fp152.py`, `quantizer_fp143.py`, `quantizer_fp107.py` | 8-bit FP(1,7,0), FP(1,6,1), FP(1,5,2), FP(1,4,3), FP(1,0,7) |
| `quantizer_fp1XX_<range>.py` | 3-bit **variance-range** quantizers: one fixed power-of-two scale for the whole network, designed for weight variance sigma^2 in `m20_p20` = [-20, 20], `m30_p30` = [-30, 30], `m35_p5` = [-35, 5], `m50_m15` = [-50, -15] dB |
| `quantizer_selector.py` | `select_best_quantizer(model)`: returns the candidate module with the highest SQNR on the weights of the trained network. Demo: `python -m quantizer.quantizer_selector` |

Scaling (argument `scale=` of every quantizer, the quantizer is `s * Q(w / s)`):
`"none"`; `"std"` per tensor, `s = sigma_tensor / target_sigma` (default target = the sigma that
maximizes the SQNR of the format); `"pow2"` same with `s` rounded to a power of two (a bit shift in
hardware); `"range"` one fixed `s` for all tensors (variance-range quantizers).
Defaults: 3-bit `std`, FP(1,7,0) / FP(1,6,1) / FP(1,5,2) `none` (their range covers any weight),
FP(1,4,3) / FP(1,0,7) `std`.

Quick test of one quantizer (MNIST 0/1, small dense network): `python -m quantizer.quantizer_fp120`

## Experiment scripts (this folder)

Run from the **project root** with the Python that has **TensorFlow 2.15 (Keras 2)**. `DIR` is the
folder with the pre-trained `.keras` models (instead of `--models-dir DIR` you can set the
environment variable `ALICE_MODELS_DIR`).

| script | what it does | output |
|---|---|---|
| `run_alice_test.py --models-dir DIR 3` | float accuracy, sigma^2 of every weight tensor, SQNR (theory / measured) and accuracy for every quantizer; which quantizer the selector picks. Ranks: `3`, `1 3 12`, `all` | `alice_quantizer_results.csv` |
| `survey_alice_weights.py --models-dir DIR` | sigma^2 [dB] of all weight tensors of all models (no evaluation, ~1 min) and the variance-range design for the real range | `alice_weight_variances.csv` |
| `run_alice_experiments.py --models-dir DIR 26 01` | A) BatchNorm re-calibration after quantization; B) sweep of `target_sigma` for FP(1,2,0) | `alice_experiments.csv` |
| `run_seed_repro.py --models-dir DIR 3 --runs 2 --epochs 5` | retrains the architecture of a model with fixed seeds; checks that the same seed gives identical results and how the accuracy before/after quantization varies across seeds. `--seeds 1,2,3`, `--bits 3\|8\|all` | `seed_repro_results.csv`, trained weights in `seed_repro_cache/` |
| `alice_loader.py` | `load_alice_model(path)`: loader for the `.keras` files (see notes) | |

`run_seed_repro.py` must get the hash seed in the same command:

```
PowerShell:  $env:PYTHONHASHSEED="0"; python experiments/fp_quantizers/run_seed_repro.py --models-dir DIR 3 --runs 2 --epochs 5
cmd:         set PYTHONHASHSEED=0 && python experiments/fp_quantizers/run_seed_repro.py --models-dir DIR 3 --runs 2 --epochs 5
```

## Notes

- **Input scaling.** The models in `cifar10_top30_best_models` were trained with the `efficient_BO`
  branch, which feeds **raw pixels 0-255** (no division by 255). With `/255` they give ~10%
  accuracy. The scripts have `NORMALIZE = False`; set it to `True` for models trained with a branch
  that normalizes.
- **Loading the `.keras` files.** `tf.keras.models.load_model` fails on these files (Keras 2.15):
  most are a `Sequential` nested in a `Sequential` ("expected 2 variables, but received 0"), and in
  the HW3 models (skip connections) the weights of some layers are stored under the name of another
  layer of the same type. `alice_loader.load_alice_model` rebuilds the network from `config.json`
  and copies the weights by name, or by shape when the names do not match.
- **Reproducibility.** The tuner seeds only the Bayesian-optimization sampler (`cfg.seed`), not
  TensorFlow, so its training is not reproducible. With `PYTHONHASHSEED=0`,
  `tf.keras.utils.set_random_seed(seed)`, `tf.config.experimental.enable_op_determinism()` and a
  seed in the shuffling and in the augmentation layers, two trainings are bit-identical
  (`run_seed_repro.py`).
- Results files and `seed_repro_cache/` are ignored by git (see `.gitignore`).
