"""
Shared helpers for the scripts in experiments/fp_quantizers.

* Puts the project root on sys.path, so that `from quantizer...` works when a script is started as
  `python experiments/fp_quantizers/<script>.py` from the project root (or from anywhere else).
* Reads the folder with the pre-trained .keras models from `--models-dir PATH` on the command line
  or from the environment variable ALICE_MODELS_DIR (the option is removed from sys.argv, so the
  scripts' own argument parsing is not affected).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (PROJECT_ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


def models_dir_from_argv():
    """Return the models folder from --models-dir PATH (or --models-dir=PATH) or $ALICE_MODELS_DIR."""
    path = None
    argv = sys.argv
    for i, a in enumerate(list(argv)):
        if a == "--models-dir" and i + 1 < len(argv):
            path = argv[i + 1]
            del argv[i:i + 2]
            break
        if a.startswith("--models-dir="):
            path = a.split("=", 1)[1]
            del argv[i]
            break
    path = path or os.environ.get("ALICE_MODELS_DIR")
    if not path:
        sys.exit("Give the folder with the .keras models: --models-dir PATH  (or set ALICE_MODELS_DIR)")
    if not os.path.isdir(path):
        sys.exit(f"Models folder not found: {path}")
    return path
