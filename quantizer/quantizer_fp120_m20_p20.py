#### FP(1,2,0): 1 sign bit, 2 exponent bits, 0 mantissa bits, variance-range quantizer.
#### Designed for weight variance sigma^2 in [-20, 20] dB: ONE fixed scale c (power of two) for all
#### tensors, chosen to maximize the mean SQNR over that range (no per-tensor statistics).
#### All logic lives in quantizer/fp_quantizer_base.py

from quantizer.fp_quantizer_base import FPQuantizerBase, run_selftest


class quantizer_module(FPQuantizerBase):
    E = 2   # exponent bits
    M = 0   # mantissa bits
    VARIANCE_RANGE = (-20, 20)   # sigma^2 in dB
    RANGE_POW2 = True             # False -> c can be any real number


if __name__ == "__main__":
    ### PyCharm: working directory = project root, or from the project root:
    ###   python -m quantizer.quantizer_fp120_m20_p20
    run_selftest(quantizer_module)
