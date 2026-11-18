#### 8-bit floating-point quantizer FP(1,0,7): 1 sign bit, 0 exponent bits, 7 mantissa bits (128 levels in [1, 2)).
#### Range [1, 2) only (no zero level): per-tensor scaling is mandatory; even then the SQNR is low (~5.8 dB).
#### All logic lives in quantizer/fp_quantizer_base.py

from quantizer.fp_quantizer_base import FPQuantizerBase, run_selftest


class quantizer_module(FPQuantizerBase):
    E = 0   # exponent bits
    M = 7   # mantissa bits
    SCALE = "std"   # default scaling mode ("none" | "std" | "pow2"); can be overridden with scale=...


if __name__ == "__main__":
    ### PyCharm: working directory = project root, or from the project root:
    ###   python -m quantizer.quantizer_fp107
    run_selftest(quantizer_module)
