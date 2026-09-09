# exp_config.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from pathlib import Path
import os, threading, yaml
import datetime

# ---------------- Dot-dict semplice ----------------
class DotDict(dict):
    __getattr__ = dict.get
    def __setattr__(self, k, v): self[k] = v
    def freeze(self):
        for k, v in list(self.items()):
            if isinstance(v, dict) and not isinstance(v, DotDict):
                self[k] = DotDict(v).freeze()
        return self

# ---------------- Schema with defaults (identical to the symbolic_tuner.py parser) ----------------
@dataclass
class ConfigSchema:
    backend: str = "tf"                          # tf | torch
    eval: int = 300                              # Max number of evaluations
    early_stop: int = 30                         # Early stopping patience
    epochs: int = 2                              # Epochs for training
    mod_list: List[str] = field(default_factory=list)  # List of active modules
    dataset: str = "light"                       # Dataset name
    name: str = "experiment"                     # Experiment name
    seed: int = 42                               # Random seed
    quantization: bool = False                   # Quantize the network
    verbose: int = 2                             # Verbosity level (0: silent, 1: space, 2: space+model)
    w_flops: float = 0.33                        # Weight Flops loss
    w_HW: float = 0.33                           # Weight HW loss
    lacc: float = 0.10                           # Accuracy loss threshold (underfitting)
    flops_th: int = 150000000                    # Max number of FLOPS
    nparams_th: int = 2500000                    # Max number of PARAMS
    opt: str = "filtered"                        # Optimizer type (standard | filtered | basic | RS | RS_ruled)
    # Gesture-specific parameters
    frames: int = 16                             # fwdPass: max sequence length (pad/truncate target); hybrid/depth: number of frames
    delta_t: int = 200000                        # fwdPass: microseconds per frame when framing with a fixed time window
    mode: str = "fwdPass"                        # Experiment mode (fwdPass, depth, hybrid)
    channels: int = 2                            # Number of channels for the dataset
    polarity: str = "both"                       # Polarity for event-based datasets (both, sum, sub, drop)

# ---------------- Validation (same rules as the parser) ----------------
_VALID_MODULES = {"hardware_module", "flops_module"}
_VALID_OPT = {"standard", "filtered", "basic", "RS", "RS_ruled"}
_VALID_BACKENDS = {"tf", "torch"}
_VALID_DATASETS = {"light", "beans", "cifar10", "cifar100", "mnist", "tinyimagenet", "gesture", "roigesture_coords", "roigesture_matrix", "roigesture_3D", "cca", "cim"}

def create_config_file(exp_dir: str | Path, overrides: Optional[Dict[str, Any]] = None) -> Path:
    """
    Create (or overwrite) a config.yaml file in the experiment directory.
    Uses schema defaults and applies the overrides passed as a dict.
    """
    p = Path(exp_dir).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    cfg_path = p / "config.yaml"

    # defaults + override
    _validate(overrides or {})
    schema = ConfigSchema()
    base = {
        "backend": schema.backend,
        "eval": schema.eval,
        "early_stop": schema.early_stop,
        "epochs": schema.epochs,
        "mod_list": schema.mod_list,
        "dataset": schema.dataset,
        "name": schema.name,
        "seed": schema.seed,
        "quantization": schema.quantization,
        "verbose": schema.verbose,
        "w_flops": schema.w_flops,
        "w_HW": schema.w_HW,
        "lacc": schema.lacc,
        "flops_th": schema.flops_th,
        "nparams_th": schema.nparams_th,
        "opt": schema.opt,
        "frames": schema.frames,
        "delta_t": schema.delta_t,
        "mode": schema.mode,
        "channels": schema.channels,
        "polarity": schema.polarity,
    }
    if overrides:
        base.update(overrides)

    # timestamp useful for traceability
    base["created_at"] = datetime.datetime.now().isoformat(timespec="seconds")

    # save YAML
    import yaml
    with open(cfg_path, "w") as f:
        yaml.safe_dump(base, f, sort_keys=False, allow_unicode=True)

    print(f"[exp_config] Created config.yaml at: {cfg_path}")
    return cfg_path

def _validate(d: Dict[str, Any]) -> None:
    # mod_list
    mods = d.get("mod_list") or []
    if isinstance(mods, str):  # In case it's a string, convert to list
        mods = [mods]
    bad = [m for m in mods if m not in _VALID_MODULES]
    if bad:
        raise ValueError(f"Invalid module(s) {bad}. Choose from: {sorted(_VALID_MODULES)}")
    
    # opt
    opt = d.get("opt", "filtered")
    if opt not in _VALID_OPT:
        print(f"WARNING: Invalid opt '{opt}'. Choose from: {sorted(_VALID_OPT)}. Set to 'filtered'")
        d['opt'] = 'filtered'
    
    # backend
    backend = d.get("backend", "tf")
    if backend not in _VALID_BACKENDS:
        print(f"WARNING: Invalid backend '{backend}'. Choose from: {sorted(_VALID_BACKENDS)}. Set to 'tf'")
        d['backend'] = 'tf'
    
    # dataset
    dataset = d.get("dataset", "light").lower().replace("-", "")
    if dataset not in _VALID_DATASETS:
        print(f"WARNING: Invalid dataset '{dataset}'. Choose from: {sorted(_VALID_DATASETS)}. Set to 'light'")
        d['dataset'] = 'light'
    
    # mode (for gesture dataset)
    mode = d.get("mode", "fwdPass")
    valid_modes = {"fwdPass", "depth", "hybrid"}
    if mode not in valid_modes:
        print(f"WARNING: Invalid mode '{mode}'. Choose from: {sorted(valid_modes)}. Set to 'fwdPass'")
        d['mode'] = 'fwdPass'
    
    # polarity (for event-based datasets)
    polarity = d.get("polarity", "both")
    valid_polarities = {"both", "sum", "sub", "drop"}
    if polarity not in valid_polarities:
        print(f"WARNING: Invalid polarity '{polarity}'. Choose from: {sorted(valid_polarities)}. Set to 'both'")
        d['polarity'] = 'both'

# ---------------- Loader + discovery ----------------
_ENV_KEY = "EXP_CONFIG"   # set this to point to the run's config file
_lock = threading.Lock()
_cached_path: Optional[Path] = None
_cached_mtime: Optional[float] = None
_cached_cfg: Optional[DotDict] = None

def _discover_config_path() -> Path:
    # 1) ENV EXP_CONFIG
    env = os.getenv(_ENV_KEY)
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            return p
    raise FileNotFoundError("config.yaml not found. Set ENV EXP_CONFIG or run from inside the experiment directory.")

def _read_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a YAML mapping (dict).")
    return data

def _apply_defaults(d: Dict[str, Any]) -> Dict[str, Any]:
    """Fill in with schema defaults."""
    schema = ConfigSchema()
    merged = {
        "backend": d.get("backend", schema.backend),
        "eval": d.get("eval", schema.eval),
        "early_stop": d.get("early_stop", schema.early_stop),
        "epochs": d.get("epochs", schema.epochs),
        "mod_list": d.get("mod_list", schema.mod_list),
        "dataset": d.get("dataset", schema.dataset),
        "name": d.get("name", schema.name),
        "seed": d.get("seed", schema.seed),
        "quantization": d.get("quantization", schema.quantization),
        "verbose": d.get("verbose", schema.verbose),
        "w_flops": d.get("w_flops", schema.w_flops),
        "w_HW": d.get("w_HW", schema.w_HW),
        "lacc": d.get("lacc", schema.lacc),
        "flops_th": d.get("flops_th", schema.flops_th),
        "nparams_th": d.get("nparams_th", schema.nparams_th),
        "opt": d.get("opt", schema.opt),
        "frames": d.get("frames", schema.frames),
        "delta_t": d.get("delta_t", schema.delta_t),
        "mode": d.get("mode", schema.mode),
        "channels": d.get("channels", schema.channels),
        "polarity": d.get("polarity", schema.polarity),
    }
    return merged

def load_cfg_from_path(path: str | Path, *, validate: bool = True) -> DotDict:
    p = Path(path).expanduser().resolve()
    d = _read_yaml(p)
    d = _apply_defaults(d)
    if validate:
        _validate(d)
    return DotDict(d).freeze()

def load_cfg(force: bool = False) -> DotDict:
    """Load (and cache) the 'active' config according to discovery rules."""
    global _cached_path, _cached_mtime, _cached_cfg
    with _lock:
        path = _discover_config_path()
        mtime = path.stat().st_mtime
        if force or _cached_cfg is None or _cached_path != path or (_cached_mtime and mtime > _cached_mtime):
            d = _read_yaml(path)
            d = _apply_defaults(d)
            _validate(d)
            _cached_cfg = DotDict(d).freeze()
            _cached_path = path
            _cached_mtime = mtime
        return _cached_cfg

def set_active_config(path: str | Path) -> None:
    """Set (for the current process) the config file to use."""
    p = Path(path).expanduser().resolve()
    os.environ[_ENV_KEY] = str(p)


def reload_cfg() -> DotDict:
    return load_cfg(force=True)