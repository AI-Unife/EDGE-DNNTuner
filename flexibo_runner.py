from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

try:
    from skopt.space import Categorical, Integer, Real, Space
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'scikit-optimize'. Install project requirements before running "
        "FlexiBO experiments: pip install -r requirements.txt"
    ) from exc

from components.controller import controller
from components.objFunction import ObjectiveWrapper
from components.search_space import search_space
from exp_config import create_config_file, load_cfg, set_active_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a FlexiBO-style BO baseline on the project RS search space."
    )
    parser.add_argument("--backend", type=str, default="tf", choices=["tf", "torch"])
    parser.add_argument("--eval", type=int, default=1000, help="Number of architectures to evaluate")
    parser.add_argument("--early_stop", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--mod_list", nargs="+", default=[])
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--name", type=str, default="flexibo_experiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantization", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--w_flops", type=float, default=0.3)
    parser.add_argument("--w_HW", type=float, default=0.33)
    parser.add_argument("--lacc", type=float, default=0.30)
    parser.add_argument("--flops_th", type=int, default=150000000)
    parser.add_argument("--nparams_th", type=int, default=15000000000)

    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--mode", type=str, default="fwdPass", choices=["fwdPass", "depth", "hybrid"])
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--polarity", type=str, default="both", choices=["both", "sum", "sub", "drop"])

    parser.add_argument("--init_random", type=int, default=10, help="Random evaluations before fitting surrogate")
    parser.add_argument("--candidate_pool", type=int, default=512, help="Random candidate configurations to score")
    parser.add_argument("--surrogate", type=str, default="GP", choices=["GP", "RF"])
    parser.add_argument("--beta", type=float, default=1.0, help="Uncertainty weight in FlexiBO LCB acquisition")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from algorithm_logs/flexibo_history.csv if it exists.",
    )
    parser.add_argument(
        "--max_new_evals",
        type=int,
        default=None,
        help="Maximum number of new architectures to evaluate in this process. Useful for Slurm chunks.",
    )
    return parser.parse_args()


def create_experiment_folders(exp_name: str) -> None:
    base_path = Path(exp_name)
    required_dirs = [
        "Model",
        "database",
        "log_folder",
        "algorithm_logs",
        "dashboard",
        "dashboard/model",
        "symbolic",
    ]
    base_path.mkdir(parents=True, exist_ok=True)
    for folder in required_dirs:
        (base_path / folder).mkdir(parents=True, exist_ok=True)


def copy_symbolic_files(exp_name: str) -> None:
    src_dir = Path("./symbolic_base")
    dst_dir = Path(exp_name) / "symbolic"
    if not src_dir.is_dir():
        return
    for src_path in src_dir.glob("*"):
        if src_path.is_file():
            shutil.copy2(src_path, dst_dir / src_path.name)


def load_dataset(cfg):
    from components.dataset import TunerDataset

    dataset = TunerDataset()
    dataset_name = cfg.dataset.lower()

    if dataset_name == "cifar10":
        dataset.n_classes = 10
        dataset.load_hf_dataset("uoft-cs/cifar10", image_key="img", label_key="label")
        dataset.normalize_data()
    elif dataset_name == "cifar100":
        dataset.n_classes = 100
        dataset.load_hf_dataset("uoft-cs/cifar100", image_key="img", label_key="fine_label")
        dataset.normalize_data()
    elif dataset_name == "mnist":
        dataset.load_mnist()
    elif dataset_name in {"cifar10_light", "light_cifar", "light"}:
        dataset.load_light_cifar()
    elif dataset_name == "gesture":
        dataset.load_gesture()
    elif "roigesture" in dataset_name:
        dataset.load_roi_gesture()
    elif dataset_name == "tinyimagenet":
        dataset.load_tiny_imagenet()
    else:
        raise ValueError(
            f"Unknown dataset: {cfg.dataset}. Supported: cifar10, cifar100, mnist, "
            "light, gesture, roigesture_matrix, roigesture_coords, tinyimagenet."
        )
    return dataset


def load_backend(cfg):
    if cfg.backend == "tf":
        from tensorflow.keras import backend as keras_backend
        from tensorflow_implementation import module_backend, neural_network

        return neural_network.NeuralNetwork, module_backend.ModuleBackend, keras_backend.clear_session

    if cfg.backend == "torch":
        from pytorch_implementation import module_backend, neural_network

        return neural_network.NeuralNetwork, module_backend.ModuleBackend, lambda: None

    raise ValueError(f"Unsupported backend: {cfg.backend}")


def normalize_for_key(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    return str(value)


def point_key(point: Sequence[Any]) -> str:
    return json.dumps([normalize_for_key(v) for v in point], sort_keys=True)


def json_params(space: Space, point: Sequence[Any]) -> str:
    params = {
        dim.name: normalize_for_key(value)
        for dim, value in zip(space.dimensions, point)
    }
    return json.dumps(params, sort_keys=True)


class FlexiboOptimizer:
    """
    FlexiBO-style black-box optimizer adapted to this project's RS search space.

    The original FlexiBO implementation builds a discrete design space from YAML
    and selects samples with surrogate uncertainty. Here we keep the same black-box
    idea, but use the DNN-Tuner skopt Space and the project ObjectiveWrapper.
    """

    def __init__(
        self,
        space: Space,
        seed: int,
        init_random: int,
        candidate_pool: int,
        surrogate: str,
        beta: float,
    ) -> None:
        self.space = space
        self.rng = np.random.RandomState(seed)
        self.seed = seed
        self.init_random = max(1, init_random)
        self.candidate_pool = max(1, candidate_pool)
        self.surrogate = surrogate
        self.beta = float(beta)
        self.points: List[List[Any]] = []
        self.scores: List[float] = []
        self.seen = set()

    def ask(self) -> List[Any]:
        if len(self.points) < self.init_random:
            return self._sample_unseen()

        candidates = self._sample_candidate_pool()
        x_train = np.asarray([self.encode(point) for point in self.points], dtype=np.float64)
        y_train = self._surrogate_scores()
        x_candidates = np.asarray([self.encode(point) for point in candidates], dtype=np.float64)

        mean, std = self._predict_region(x_train, y_train, x_candidates)
        acquisition = mean - (math.sqrt(self.beta) * std)
        return candidates[int(np.argmin(acquisition))]

    def tell(self, point: Sequence[Any], score: float) -> None:
        self.points.append(list(point))
        self.scores.append(float(score))
        self.seen.add(point_key(point))

    def _surrogate_scores(self) -> np.ndarray:
        y = np.asarray(self.scores, dtype=np.float64)
        valid = y[np.isfinite(y) & (np.abs(y) < 1e9)]
        if valid.size == 0:
            return np.zeros_like(y)

        worst_valid = float(np.max(valid))
        spread = float(np.max(valid) - np.min(valid)) or 1.0
        penalty_value = worst_valid + spread
        return np.where(np.isfinite(y) & (np.abs(y) < 1e9), y, penalty_value)

    def _sample_unseen(self) -> List[Any]:
        for _ in range(1000):
            point = self.space.rvs(n_samples=1, random_state=self.rng)[0]
            if point_key(point) not in self.seen:
                return list(point)
        return list(self.space.rvs(n_samples=1, random_state=self.rng)[0])

    def _sample_candidate_pool(self) -> List[List[Any]]:
        candidates: List[List[Any]] = []
        local_seen = set()
        attempts = 0
        while len(candidates) < self.candidate_pool and attempts < self.candidate_pool * 100:
            attempts += 1
            point = self.space.rvs(n_samples=1, random_state=self.rng)[0]
            key = point_key(point)
            if key in self.seen or key in local_seen:
                continue
            local_seen.add(key)
            candidates.append(list(point))

        while len(candidates) < self.candidate_pool:
            candidates.append(self._sample_unseen())

        return candidates

    def encode(self, point: Sequence[Any]) -> List[float]:
        encoded: List[float] = []
        for dim, value in zip(self.space.dimensions, point):
            if isinstance(dim, Categorical):
                encoded.extend(1.0 if value == category else 0.0 for category in dim.categories)
            elif isinstance(dim, Integer):
                low, high = float(dim.low), float(dim.high)
                encoded.append(0.0 if math.isclose(low, high) else (float(value) - low) / (high - low))
            elif isinstance(dim, Real):
                low, high = float(dim.low), float(dim.high)
                encoded.append(0.0 if math.isclose(low, high) else (float(value) - low) / (high - low))
            else:
                raise TypeError(f"Unsupported skopt dimension type: {type(dim)}")
        return encoded

    def _predict_region(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_candidates: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.surrogate == "RF":
            return self._predict_rf(x_train, y_train, x_candidates)
        return self._predict_gp(x_train, y_train, x_candidates)

    def _predict_gp(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_candidates: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern, WhiteKernel
        except ModuleNotFoundError as exc:
            raise RuntimeError("scikit-learn is required for the FlexiBO GP surrogate.") from exc

        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-6)
        model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=self.seed + len(self.points),
            n_restarts_optimizer=0,
        )
        model.fit(x_train, y_train)
        mean, std = model.predict(x_candidates, return_std=True)
        return np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)

    def _predict_rf(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_candidates: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ModuleNotFoundError as exc:
            raise RuntimeError("scikit-learn is required for the FlexiBO RF surrogate.") from exc

        model = RandomForestRegressor(
            n_estimators=64,
            min_samples_leaf=1,
            random_state=self.seed + len(self.points),
            n_jobs=1,
        )
        model.fit(x_train, y_train)
        tree_predictions = np.asarray(
            [tree.predict(x_candidates) for tree in model.estimators_],
            dtype=np.float64,
        )
        return tree_predictions.mean(axis=0), tree_predictions.std(axis=0)


def append_history(log_path: Path, row: Dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with log_path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "iteration",
                "score",
                "best_score",
                "elapsed_sec",
                "params_json",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def value_from_history(dim: Any, raw_value: Any) -> Any:
    if isinstance(dim, Categorical):
        for category in dim.categories:
            if normalize_for_key(category) == raw_value:
                return category
        raise ValueError(f"History value {raw_value!r} is not valid for categorical dimension {dim.name!r}.")
    if isinstance(dim, Integer):
        return int(raw_value)
    if isinstance(dim, Real):
        return float(raw_value)
    raise TypeError(f"Unsupported skopt dimension type: {type(dim)}")


def point_from_params_json(space: Space, params_json: str) -> List[Any]:
    params = json.loads(params_json)
    if not isinstance(params, dict):
        raise ValueError("History params_json must decode to a JSON object.")

    point: List[Any] = []
    for dim in space.dimensions:
        if dim.name is None:
            raise ValueError("Cannot resume: all search-space dimensions must have a name.")
        if dim.name not in params:
            raise ValueError(f"Cannot resume: missing dimension {dim.name!r} in history row.")
        point.append(value_from_history(dim, params[dim.name]))
    return point


def load_history(log_path: Path, space: Space, optimizer: FlexiboOptimizer) -> int:
    if not log_path.exists():
        return 0

    loaded = 0
    with log_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            point = point_from_params_json(space, row["params_json"])
            score = float(row["score"])
            optimizer.tell(point, score)
            loaded += 1
    return loaded


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    exp_dir = Path(args.name)
    overrides = vars(args).copy()
    overrides["opt"] = "RS"
    overrides["external_tuner"] = "FlexiBO"
    overrides["search_space_source"] = "RS"

    cfg_path = create_config_file(exp_dir, overrides=overrides)
    set_active_config(cfg_path)
    cfg = load_cfg(force=True)

    create_experiment_folders(cfg.name)
    copy_symbolic_files(cfg.name)

    dataset = load_dataset(cfg)
    neural_network_cls, module_backend_cls, clear_session_callback = load_backend(cfg)
    ctrl = controller(
        neural_network_cls,
        module_backend_cls(),
        dataset,
        clear_session_callback=clear_session_callback,
    )

    base_space = search_space().search_sp(max_block=ctrl.max_conv, max_dense=ctrl.max_fc)
    objective = ObjectiveWrapper(base_space, ctrl)
    optimizer = FlexiboOptimizer(
        space=base_space,
        seed=cfg.seed,
        init_random=args.init_random,
        candidate_pool=args.candidate_pool,
        surrogate=args.surrogate,
        beta=args.beta,
    )

    print("\nSTARTING FLEXIBO BASELINE\n")
    start_time = time.time()
    best_score = float("inf")
    history_path = Path(cfg.name) / "algorithm_logs" / "flexibo_history.csv"
    completed_evals = 0

    if args.resume:
        completed_evals = load_history(history_path, base_space, optimizer)
        if completed_evals:
            best_score = min(optimizer.scores)
            print(f"[INFO] Resumed {completed_evals} previous FlexiBO evaluations from {history_path}.")
        else:
            print("[INFO] Resume requested, but no previous FlexiBO history was found. Starting fresh.")

    target_eval = args.eval
    if args.max_new_evals is not None:
        if args.max_new_evals < 1:
            raise ValueError("--max_new_evals must be >= 1 when provided.")
        target_eval = min(args.eval, completed_evals + args.max_new_evals)

    if completed_evals >= args.eval:
        print(f"[INFO] Requested eval budget already reached: {completed_evals} / {args.eval}.")
        print(f"HISTORY -----------> {history_path}")
        return

    for iteration in range(completed_evals + 1, target_eval + 1):
        if ctrl.convergence:
            print("[INFO] Controller early stopping triggered.")
            break

        print(f"\n--- FLEXIBO ITERATION {iteration} / {args.eval} ---")
        point = optimizer.ask()
        score = float(objective.objective(point))
        optimizer.tell(point, score)
        best_score = min(best_score, score)

        append_history(
            history_path,
            {
                "iteration": iteration,
                "score": score,
                "best_score": best_score,
                "elapsed_sec": round(time.time() - start_time, 4),
                "params_json": json_params(base_space, point),
            },
        )

    total_time = time.time() - start_time
    print("\nFLEXIBO BASELINE FINISHED")
    print(f"TOTAL TIME --------> {total_time:.2f} seconds")
    print(f"BEST SCORE --------> {best_score}")
    print(f"HISTORY -----------> {history_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
