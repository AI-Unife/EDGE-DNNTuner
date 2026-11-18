#### 8-bit floating-point quantizer FP(1,6,1): 1 sign bit, 6 exponent bits, 1 mantissa bit.
#### Dynamic range 2^-31 .. 2^33 covers any weight, so the SQNR does not depend on sigma: NO scaling needed.
#### All logic lives in quantizer/fp_quantizer_base.py

from quantizer.fp_quantizer_base import FPQuantizerBase, run_selftest


class quantizer_module(FPQuantizerBase):
    E = 6   # exponent bits
    M = 1   # mantissa bits
    SCALE = "none"   # default scaling mode ("none" | "std" | "pow2"); can be overridden with scale=...


if __name__ == "__main__":
    ### PyCharm: working directory = project root, or from the project root:
    ###   python -m quantizer.quantizer_fp161
    run_selftest(quantizer_module)
