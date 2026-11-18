#### 8-bit floating-point quantizer FP(1,4,3): 1 sign bit, 4 exponent bits, 3 mantissa bits.
#### Range 2^-7 .. 2^8.9 (no zero level): weights below 0.0078 would be rounded up, so per-tensor scaling is used.
#### All logic lives in quantizer/fp_quantizer_base.py

from quantizer.fp_quantizer_base import FPQuantizerBase, run_selftest


class quantizer_module(FPQuantizerBase):
    E = 4   # exponent bits
    M = 3   # mantissa bits
    SCALE = "std"   # default scaling mode ("none" | "std" | "pow2"); can be overridden with scale=...


if __name__ == "__main__":
    ### PyCharm: working directory = project root, or from the project root:
    ###   python -m quantizer.quantizer_fp143
    run_selftest(quantizer_module)
