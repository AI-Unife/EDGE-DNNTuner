#### 3-bit floating-point quantizer FP(1,2,0): 1 sign bit, 2 exponent bits, 0 mantissa bits (no scaling)
#### levels |w| in {0.5, 1, 2, 4}. All logic lives in quantizer/fp_quantizer_base.py

from quantizer.fp_quantizer_base import FPQuantizerBase, run_selftest


class quantizer_module(FPQuantizerBase):
    E = 2   # exponent bits
    M = 0   # mantissa bits


if __name__ == "__main__":
    ### Run this file to test the quantizer (PyCharm: working directory = project root,
    ### or from a terminal in the project root: python -m quantizer.quantizer_fp120)
    run_selftest(quantizer_module)
