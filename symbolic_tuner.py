from __future__ import annotations
import argparse
import time
import sys
import os
import copy
import shutil
from typing import List
from pathlib import Path 

from skopt.space import Categorical, Space

from components.colors import colors
from components.controller import controller
from components.search_space import search_space
from components.random_search import RandomSearch
from components.objFunction import ObjectiveWrapper
from components.constraints import ConstraintsWrapper
from components.dataset import TunerDataset
from exp_config import create_config_file, set_active_config, load_cfg

# ------------------------------ filesystem -----------------------------------

def create_experiment_folders() -> None:
    """
    Ensure the experiment directory structure exists.
    """
    # Use the 'cfg.name' Path object from the main script
    base_path = Path(cfg.name)
    required_dirs = [
        "Model", "database",
        "algorithm_logs", "dashboard", "dashboard/model", "symbolic"
    ]
    
    try:
        base_path.mkdir(exist_ok=True)
        for folder in required_dirs:
            (base_path / folder).mkdir(exist_ok=True)
    except OSError as e:
        print(colors.FAIL, f"Failed to create folder: {e}", colors.ENDC)
        sys.exit(1) # This is okay for a setup script, it's a critical error.


def copy_symbolic_files() -> None:
    """
    Copy the base symbolic files into this experiment's symbolic folder.
    """
    src_dir = Path("./symbolic_base")
    dst_dir = Path(f"{cfg.name}/symbolic") # or Path(cfg.name) / "symbolic"

    if not src_dir.is_dir():
        print(colors.WARNING, f"Symbolic base folder not found at '{src_dir}'. Skipping copy.", colors.ENDC)
        return
    try:
        # Copy only files (flat copy). Avoids deep-copying and is simpler.
        for src_path in src_dir.glob('*'): # glob('*') gets files and dirs
            if src_path.is_file():
                dst_path = dst_dir / src_path.name
                shutil.copy2(src_path, dst_path)
    except OSError as e:
        print(colors.FAIL, f"Failed to copy symbolic directory: {e}", colors.ENDC)
        sys.exit(1) # Critical setup failure



# ------------------------------- helpers -------------------------------------

def print_space(space: Space) -> None:
    """Pretty print the current Space dimensions with names and bounds."""
    for i, dim in enumerate(space.dimensions):
        if isinstance(dim, Categorical):
            print(f"Dimension {i}: {dim.name} - categories: {dim.categories}")
        else:
            if dim.high != 0:
                print(f"Dimension {i}: {dim.name} - {dim}")


# ------------------------------ optimization ---------------------------------


def run_optimization(base_space: Space, first_ss: search_space, ctrl: controller, max_iter: int):
    """
    Outer optimization loop:
      - initialize (RS or BO),
      - iteratively let the controller diagnose and (optionally) mutate the space,
      - reuse past evaluations when the space is unchanged,
      - checkpoint after each evaluation.
    """
    # --- 1. Initialization ---
    callback = None # CheckpointSaver(ckpt_path, compress=9)
    no_rules = ["RS", "standard"]
    with_rules = ["filtered", "RS_ruled", "basic"]
    use_filter = (cfg.opt == "filtered")

    # const_space is the *dynamic* space, which may be mutated by rules.
    # base_space is the *static* reference, which we expand if new dims are found.
    const_space = search_space().search_sp(max_block=ctrl.max_conv, max_dense=ctrl.max_fc)
    
    # x0, y0 hold the history of points. Starts empty.
    x0, y0 = [], []
    res = None
    
    # Initialize the optimizer instance if using custom RandomSearch
    random_search_optimizer = None
    if "RS" in cfg.opt:
        random_search_optimizer = RandomSearch(random_state=cfg.seed, total_iter=max_iter)

    # --- 2. Main Optimization Loop ---
    # We use ctrl.iter (which starts at 1) and loop *while* it's <= max_iter
    while ctrl.iter <= max_iter and not ctrl.convergence:
        print(colors.MAGENTA, f"--- ITERATION {ctrl.iter} ---", colors.ENDC)

        # (Re)create the objective and constraint functions for this iteration.
        # This ensures they capture the *current* const_space
        obj_fn = ObjectiveWrapper(const_space, ctrl)
        const_fn = ConstraintsWrapper(const_space)

        if cfg.verbose > 0:
            print("Actual constrained space for this iteration:")
            print_space(const_space)
        #     print("Actual base space for this iteration:")
        #     print_space(base_space)

        # --- 3. Run One Step of Optimization ---
        try:
            if "RS" in cfg.opt:
                # Random search: draw one more sample/eval
                res = random_search_optimizer(obj_fn.objective, const_space,
                                              callback=callback
                                              )
            else:
                # Bayesian Optimization: one call.
                # If x0, y0 are empty, it's a "cold-start" (n_random_starts=1)
                # If x0, y0 are present, it's a "warm-start" (n_random_starts=0)
                is_warm_start = len(y0) > 0
                res = gp_minimize(
                    obj_fn.objective,
                    base_space,  
                    x0=list(x0) if is_warm_start else None,
                    y0=list(y0) if is_warm_start else None,
                    acq_func="EI",
                    n_calls=1,
                    n_random_starts= 0 if is_warm_start else 1,
                    random_state=cfg.seed,
                    callback=callback,
                    space_constraint=const_fn.apply_constraints if use_filter else None
                )
        except Exception as e:
            # Fallback: If gp_minimize fails (e.g., matrix inversion),
            # try a random step to get new data.
            print(colors.FAIL, f"Optimization error: {e}. Retrying with a random sample.", colors.ENDC)
            if "RS" in cfg.opt:
                res = random_search_optimizer(obj_fn.objective, base_space, callback=callback)
            else:
                # Use standard gp_minimize with n_random_starts=1 to get a new point
                res = gp_minimize(
                    obj_fn.objective, base_space,
                    acq_func="EI", n_calls=1, n_random_starts=1,
                    random_state=cfg.seed, callback=callback,
                    space_constraint=const_fn.apply_constraints if use_filter else None
                )
        
        # Update our history of points
        x0, y0 = res.x_iters, res.func_vals

        # --- 4. Diagnosis and Space Update ---
        # Ask the controller to propose the *next* space based on this iteration
        if cfg.opt in no_rules:
            # 'no_rules' means the space never changes
            next_space = copy.deepcopy(base_space)
        else:
            # 'with_rules' means the controller diagnoses and mutates the space
            next_space = ctrl.diagnosis(const_space)
            # Expand the base_space if the controller added new dimensions
            if cfg.opt == 'basic':
                base_space = next_space
            else:
                base_space = first_ss.expand_space(base_space, next_space)

        # --- 5. Check for Space Change (This is the critical bug fix) ---
        if len(next_space.dimensions) != len(const_space.dimensions):
            # Space dimensions have changed!
            # We must reset the optimizer and start cold (no x0, y0).
            print(colors.WARNING, "Search space dimensions have changed! Restarting optimizer.", colors.ENDC)
            x0, y0 = [], [] # Reset warm-start data
        
        # The space for the *next* iteration is the one just diagnosed
        const_space = copy.deepcopy(next_space)

        # Note: ctrl.iter is incremented *inside* ctrl.training()
    
    print(colors.OKGREEN, "\nOptimization loop finished.", colors.ENDC)
    return res

# ---------------------------------- main -------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the Symbolic DNN Tuner configuration.

    Returns:
        argparse.Namespace containing all configuration values.
    """
    parser = argparse.ArgumentParser(
        description="Symbolic DNN Tuner Configuration"
    )
    parser.add_argument("--backend", type=str, default="tf",
                        choices=["tf", "torch"],
                        help="Backend framework to use (TensorFlow or PyTorch)")
    parser.add_argument("--eval", type=int, default=300,
                        help="Max number of evaluations")
    parser.add_argument("--early_stop", type=int, default=30,
                        help="Early stopping patience")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Epochs for training")
    parser.add_argument(
        "--mod_list", nargs="+", default=[],
        help="List of active modules (e.g., hardware_module flops_module)"
    )
    parser.add_argument("--dataset", type=str, default="light",
                        help="Dataset name")
    parser.add_argument("--name", type=str, default="experiment",
                        help="Experiment name")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument('--quantization', action='store_true',
                        help='quantize the network')
    parser.add_argument("--verbose", type=int, default=2, 
                        help="Verbosity level (0: silent, 1: print space, 2: print space and model summary)")
    parser.add_argument('--w_flops', type=float, default=0.3,
                        help="Weight Flops loss")
    parser.add_argument('--w_HW', type=float, default=0.33,
                        help="Weight HW loss")
    parser.add_argument('--lacc', type=float, default=0.30,
                        help="if 1-acc>lacc --> Underfitting")
    parser.add_argument('--flops_th', type=int, default=15000000000,
                         help="Max number of FLOPS")
    parser.add_argument('--nparams_th', type=int, default=150000000,
                        help="Max number of PARAMS")
    parser.add_argument(
        "--opt", type=str, default="filtered",
        choices=["standard", "filtered", "basic", "RS", "RS_ruled"],
        help="Optimizer type for the analysis"
    )   
    
    #### Gesture option
    parser.add_argument("--frames", type=int, default=16,
                        help="Number of frames for gesture dataset")
    parser.add_argument("--mode", type=str, default="fwdPass",
                        choices=["fwdPass", "depth", "hybrid"],
                        help="Experiment mode (fwdPass, depth, hybrid)")
    parser.add_argument("--channels", type=int, default=2,
                        help="Number of channels for the dataset")
    parser.add_argument("--polarity", type=str, default="both",
                        choices=["both", "sum", "sub", "drop"],
                        help="Polarity for event-based datasets")


    args = parser.parse_args()

    # Validate module list
    valid_modules = {"hardware_module", "flops_module"}
    for mod in args.mod_list:
        if mod not in valid_modules:
            parser.error(
                f"Invalid module '{mod}'. Choose from: {', '.join(valid_modules)}"
            )

    return args



if __name__ == "__main__":
    # Optional cosmetic for Windows terminals (enables ANSI colors)
    if os.name == 'nt':
        os.system("")


    # --- 1. Configuration Setup ---
    args = parse_args()
    
    exp_dir = Path(args.name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    cfg_path = create_config_file(exp_dir, overrides=args.__dict__)
    set_active_config(cfg_path)
    cfg = load_cfg(force=True)
    
    # Dynamically import the correct gp_minimize based on config
    if cfg.opt == 'basic':
        from skopt import gp_minimize
    else:
        from components.bo.gp import gp_minimize
    
    if cfg.backend == "tf":
        from tensorflow_implementation import module_backend, neural_network
        from tensorflow.keras import backend as K
        clear_session_callback = K.clear_session
    elif cfg.backend == "torch":
        from pytorch_implementation import module_backend, neural_network
        clear_session_callback = lambda: None
    else:
        raise ValueError(f"Unsupported backend: {cfg.backend}")

    # --- 2. Filesystem and Dataset Setup ---
    create_experiment_folders() # Uses new pathlib version
    copy_symbolic_files()   # Uses new pathlib version

    # Load dataset based on command-line argument
    dataset = TunerDataset()
    dataset_name = cfg.dataset.lower()
    
    if dataset_name == "cifar10":
        dataset.load_cifar_10()
    elif dataset_name == "cifar100":
        dataset.load_cifar_100()
    elif dataset_name == "mnist":
        dataset.load_mnist()
    elif dataset_name == "cifar10_light" or dataset_name == "light_cifar" or dataset_name == "light":
        dataset.load_light_cifar()
    elif dataset_name == "gesture":
        dataset.load_gesture()
    elif "roigesture" in dataset_name:
        dataset.load_roi_gesture()
    elif dataset_name == "tinyimagenet":
        dataset.load_tiny_imagenet()
    elif dataset_name == "cca":
        dataset.load_cca()
    elif dataset_name == "cim":
        dataset.load_cim()
    elif dataset_name == "beans":
        dataset.load_beans()
    else:
        print(f"Unknown dataset: {cfg.dataset}. Supported: cifar10, cifar100, mnist, beans, light, gesture, roigesture_matrix, roigesture_coords, cca, cim.")
        exit(1)

    # --- 3. Controller and Space Setup ---
    neural_network_cls = neural_network.NeuralNetwork
    module_backend_cls = module_backend.ModuleBackend
    
    ctrl = controller(
            neural_network_cls,
            module_backend_cls(),
            dataset,
            clear_session_callback=clear_session_callback,
        )

    # The 'first_ss' object is used to manage and expand the space
    first_ss = search_space()
    first_space = first_ss.search_sp(max_block=ctrl.max_conv, max_dense=ctrl.max_fc)
    
    # print(colors.MAGENTA, "|  ----------- BASIC SEARCH SPACE ----------  |\n", colors.ENDC)
    # print_space(first_space) # Use helper to print

    # --- 4. Run Optimization ---
    start_time = time.time()
    print(colors.OKGREEN, "\nSTARTING ALGORITHM \n", colors.ENDC)
    
    res = run_optimization(
        base_space=first_space, 
        first_ss=first_ss, 
        ctrl=ctrl, 
        max_iter=cfg.eval
    )
    
    # Remove temo file in dashboard
    shutil.rmtree(Path(cfg.name)/"dashboard")
    
    end_time = time.time()
    print(colors.OKGREEN, "\nALGORITHM FINISHED \n", colors.ENDC)
    print(colors.CYAN, f"\nTOTAL TIME --------> {end_time - start_time:.2f} seconds\n", colors.ENDC)