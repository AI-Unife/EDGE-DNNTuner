#### 3-bit floating-point quantizer FP(1,1,1): 1 sign bit, 1 exponent bit, 1 mantissa bit (no scaling)
#### levels |w| in {1, 1.5, 2, 3}. All logic lives in quantizer/fp_quantizer_base.py

from quantizer.fp_quantizer_base import FPQuantizerBase, run_selftest


class quantizer_module(FPQuantizerBase):
    E = 1   # exponent bits
    M = 1   # mantissa bits


if __name__ == "__main__":
    ### Run this file to test the quantizer (PyCharm: working directory = project root,
    ### or from a terminal in the project root: python -m quantizer.quantizer_fp111)
    run_selftest(quantizer_module)
