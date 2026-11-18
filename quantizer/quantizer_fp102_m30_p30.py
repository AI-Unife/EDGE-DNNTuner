#### FP(1,0,2): 1 sign bit, 0 exponent bits, 2 mantissa bits, variance-range quantizer.
#### Designed for weight variance sigma^2 in [-30, 30] dB: ONE fixed scale c (power of two) for all
#### tensors, chosen to maximize the mean SQNR over that range (no per-tensor statistics).
#### All logic lives in quantizer/fp_quantizer_base.py

from quantizer.fp_quantizer_base import FPQuantizerBase, run_selftest


class quantizer_module(FPQuantizerBase):
    E = 0   # exponent bits
    M = 2   # mantissa bits
    VARIANCE_RANGE = (-30, 30)   # sigma^2 in dB
    RANGE_POW2 = True             # False -> c can be any real number


if __name__ == "__main__":
    ### PyCharm: working directory = project root, or from the project root:
    ###   python -m quantizer.quantizer_fp102_m30_p30
    run_selftest(quantizer_module)
