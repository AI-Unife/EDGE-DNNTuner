from components.colors import colors
from components.controller import controller
from exp_config import load_cfg
from skopt.space import Space
from typing import List
from pathlib import Path


class ObjectiveWrapper:
    """
    Helper to convert a sampled vector from skopt Space to a `dict` for the controller.
    """
    def __init__(self, space: Space, ctrl: controller):
        self.search_space = space
        self.controller = ctrl
        self.exp_cfg = load_cfg()

    def objective(self, params: List) -> float:
        """
        Convert vector `params` to {name: value}, log it, train once, and return the score.
        """
        # 1. Convert list of parameters to a dictionary
        space_dict = {dim.name: val for dim, val in zip(self.search_space.dimensions, params)}
        print("Chosen point:", space_dict)

        # 2. Log the chosen parameters
        log_path = Path(self.exp_cfg.name) / "algorithm_logs" / "hyper-neural.txt"
        try:
            with open(log_path, "a") as f:
                f.write(str(space_dict) + "\n")
        except OSError as e:
            print(colors.WARNING, f"Could not write to log file {log_path}: {e}", colors.ENDC)
        
        # 3. Run the training and get the score
        score = self.controller.training(space_dict)
        score = float(score) if score is not None else None
        return score