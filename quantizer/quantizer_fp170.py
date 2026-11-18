#### 8-bit floating-point quantizer FP(1,7,0): 1 sign bit, 7 exponent bits, 0 mantissa bits (128 power-of-two levels).
#### Dynamic range 2^-63 .. 2^64 covers any weight, so the SQNR does not depend on sigma: NO scaling needed.
#### All logic lives in quantizer/fp_quantizer_base.py

from quantizer.fp_quantizer_base import FPQuantizerBase, run_selftest


class quantizer_module(FPQuantizerBase):
    E = 7   # exponent bits
    M = 0   # mantissa bits
    SCALE = "none"   # default scaling mode ("none" | "std" | "pow2"); can be overridden with scale=...


if __name__ == "__main__":
    ### PyCharm: working directory = project root, or from the project root:
    ###   python -m quantizer.quantizer_fp170
    run_selftest(quantizer_module)
