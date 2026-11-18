#### 3-bit floating-point quantizer FP(1,0,2): 1 sign bit, 0 exponent bits, 2 mantissa bits (no scaling)
#### levels |w| in {1, 1.25, 1.5, 1.75}. All logic lives in quantizer/fp_quantizer_base.py

from quantizer.fp_quantizer_base import FPQuantizerBase, run_selftest


class quantizer_module(FPQuantizerBase):
    E = 0   # exponent bits
    M = 2   # mantissa bits


if __name__ == "__main__":
    ### Run this file to test the quantizer (PyCharm: working directory = project root,
    ### or from a terminal in the project root: python -m quantizer.quantizer_fp102)
    run_selftest(quantizer_module)
