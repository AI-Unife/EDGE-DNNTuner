from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

FLEXIBO_PATH = Path(__file__).resolve().parent / "othertunerdependencies" / "flexibo" / "FlexiBO"
if FLEXIBO_PATH.exists():
    sys.path.insert(0, str(FLEXIBO_PATH))

try:
    from src.sampling import Sampling as OriginalFlexiboSampling
    from src.surrogate_model import GPSurrogateModel as OriginalFlexiboGPSurrogateModel
    from src.surrogate_model import RFSurrogateModel as OriginalFlexiboRFSurrogateModel
    from src.utils import Utils as OriginalFlexiboUtils
except ModuleNotFoundError:
    OriginalFlexiboGPSurrogateModel = None
    OriginalFlexiboRFSurrogateModel = None
    OriginalFlexiboSampling = None
    OriginalFlexiboUtils = None

try:
    from skopt.space import Categorical, Integer, Real, Space
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'scikit-optimize'. Install project requirements before running "
        "FlexiBO experiments: pip install -r requirements.txt"
    ) from exc

from components.controller import controller
from components.search_space import search_space
from exp_config import create_config_file, load_cfg, set_active_config


PENALTY_SCORE = 1e10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a multi-objective FlexiBO-style baseline on the project RS search space."
    )
    parser.add_argument("--backend", type=str, default="tf", choices=["tf", "torch"])
    parser.add_argument("--eval", type=int, default=1000, help="Number of accuracy/training evaluations")
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

    parser.add_argument("--init_random", type=int, default=10, help="Initial evaluations measuring both objectives")
    parser.add_argument("--candidate_pool", type=int, default=512, help="Random candidate configurations to score")
    parser.add_argument("--surrogate", type=str, default="GP", choices=["GP", "RF"])
    parser.add_argument("--beta", type=float, default=1.0, help="Uncertainty weight in FlexiBO regions")
    parser.add_argument("--accuracy_cost", type=float, default=1.0, help="Relative cost of training/accuracy")
    parser.add_argument("--flops_cost", type=float, default=0.05, help="Relative cost of FLOPs-only measurement")
    parser.add_argument(
        "--max_flops_only_streak",
        type=int,
        default=5,
        help="Force an accuracy evaluation after this many consecutive FLOPs-only events.",
    )
    parser.add_argument(
        "--flops_scale",
        type=float,
        default=None,
        help="Scale used for normalized FLOPs objective. Defaults to --flops_th.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from algorithm_logs/flexibo_history.csv if it exists.",
    )
    parser.add_argument(
        "--max_new_evals",
        type=int,
        default=None,
        help="Maximum number of new accuracy/training evaluations in this process. Useful for Slurm chunks.",
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
    return json.dumps(
        {dim.name: normalize_for_key(value) for dim, value in zip(space.dimensions, point)},
        sort_keys=True,
    )


def point_to_params(space: Space, point: Sequence[Any]) -> Dict[str, Any]:
    return {dim.name: value for dim, value in zip(space.dimensions, point)}


@dataclass
class Observation:
    point: List[Any]
    error: float | None = None
    flops_norm: float | None = None
    accuracy: float | None = None
    flops: float | None = None
    params: float | None = None
    score: float | None = None


class FlexiboEvaluator:
    def __init__(self, space: Space, ctrl: controller, flops_scale: float) -> None:
        self.space = space
        self.ctrl = ctrl
        self.flops_scale = float(flops_scale) or 1.0

    def _params(self, point: Sequence[Any]) -> Dict[str, Any]:
        params = point_to_params(self.space, point)
        log_path = Path(self.ctrl.exp_cfg.name) / "algorithm_logs" / "hyper-neural.txt"
        with log_path.open("a") as f:
            f.write(str(params) + "\n")
        return params

    def evaluate_flops_only(self, point: Sequence[Any]) -> Dict[str, float | None]:
        params = self._params(point)
        print("Chosen point for FLOPs-only:", params)

        if self.ctrl.clear_session_callback:
            self.ctrl.clear_session_callback()

        self.ctrl.params = params
        self.ctrl.set_data_augmentation(params.get("data_augmentation", False))
        self.ctrl.set_reg_l2(params.get("reg_l2", False))
        self.ctrl.set_residual(params.get("skip_connection", False))
        self.ctrl.nn.build_network(params, self.ctrl.layer_x_block)

        flops = float(self.ctrl.nn.flops or 0.0)
        nparams = float(self.ctrl.nn.nparams or 0.0)
        self._append_scalar("flexibo_flops_only_report.txt", f"{nparams} {flops}")
        return {
            "flops": flops,
            "params": nparams,
            "flops_norm": flops / self.flops_scale,
        }

    def evaluate_accuracy(self, point: Sequence[Any]) -> Dict[str, float | None]:
        params = self._params(point)
        print("Chosen point for accuracy:", params)
        score = float(self.ctrl.training(params))

        accuracy = None
        error = None
        if self.ctrl.scoreNN is not None:
            accuracy = float(self.ctrl.scoreNN[1])
            error = 1.0 - accuracy
        else:
            error = PENALTY_SCORE

        flops = float(self.ctrl.nn.flops or 0.0)
        nparams = float(self.ctrl.nn.nparams or 0.0)
        return {
            "score": score,
            "accuracy": accuracy,
            "error": error,
            "flops": flops,
            "params": nparams,
            "flops_norm": flops / self.flops_scale,
        }

    def _append_scalar(self, filename: str, value: str) -> None:
        path = Path(self.ctrl.exp_cfg.name) / "algorithm_logs" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(value + "\n")


class FlexiboOptimizer:
    """
    Multi-objective FlexiBO-style optimizer for RS.

    Objective O1 is validation error (1 - accuracy), which requires training.
    Objective O2 is normalized FLOPs, which is cheap and can be measured by
    building the model only. The acquisition selects both a configuration and
    the next objective to measure, using uncertainty shrinkage per objective cost.
    """

    def __init__(
        self,
        space: Space,
        seed: int,
        init_random: int,
        candidate_pool: int,
        surrogate: str,
        beta: float,
        accuracy_cost: float,
        flops_cost: float,
    ) -> None:
        self.space = space
        self.rng = np.random.RandomState(seed)
        self.seed = seed
        self.init_random = max(1, init_random)
        self.candidate_pool = max(1, candidate_pool)
        self.surrogate = surrogate
        self.beta = float(beta)
        self.accuracy_cost = max(float(accuracy_cost), 1e-12)
        self.flops_cost = max(float(flops_cost), 1e-12)
        self.observations: Dict[str, Observation] = {}
        self.order: List[str] = []
        self.original_sampling = None
        self.original_utils = None
        if OriginalFlexiboSampling is not None and OriginalFlexiboUtils is not None:
            self.original_sampling = OriginalFlexiboSampling(0, 1, self.accuracy_cost, self.flops_cost)
            self.original_utils = OriginalFlexiboUtils(0, 1)
        self.original_gp = OriginalFlexiboGPSurrogateModel() if OriginalFlexiboGPSurrogateModel is not None else None
        self.original_rf = OriginalFlexiboRFSurrogateModel() if OriginalFlexiboRFSurrogateModel is not None else None

    @property
    def accuracy_evals(self) -> int:
        return sum(1 for obs in self.observations.values() if obs.error is not None)

    def tell(
        self,
        point: Sequence[Any],
        objective: str,
        result: Dict[str, float | None],
    ) -> Observation:
        key = point_key(point)
        if key not in self.observations:
            self.observations[key] = Observation(point=list(point))
            self.order.append(key)
        obs = self.observations[key]

        if result.get("flops_norm") is not None:
            obs.flops_norm = float(result["flops_norm"])
        if result.get("flops") is not None:
            obs.flops = float(result["flops"])
        if result.get("params") is not None:
            obs.params = float(result["params"])

        if objective in {"accuracy", "both"}:
            if result.get("accuracy") is not None:
                obs.accuracy = float(result["accuracy"])
            if result.get("error") is not None:
                obs.error = float(result["error"])
            if result.get("score") is not None:
                obs.score = float(result["score"])
        return obs

    def ask(self) -> tuple[List[Any], str]:
        if self.accuracy_evals < self.init_random:
            return self._sample_new_point(), "both"

        candidates = self._candidate_pool()
        means, stds = self._predict_objectives(candidates)
        original_choice = self._ask_with_original_flexibo(candidates, means, stds)
        if original_choice is not None:
            return original_choice

        pareto_indices = self._optimistic_pareto_indices(means, stds)
        if not pareto_indices:
            pareto_indices = list(range(len(candidates)))

        best_choice = None
        best_value = -float("inf")
        for idx in pareto_indices:
            point = candidates[idx]
            key = point_key(point)
            obs = self.observations.get(key)
            err_unmeasured = obs is None or obs.error is None
            flops_unmeasured = obs is None or obs.flops_norm is None

            if err_unmeasured:
                value = (2.0 * math.sqrt(self.beta) * stds["error"][idx]) / self.accuracy_cost
                if value > best_value:
                    best_choice = (point, "accuracy")
                    best_value = value
            if flops_unmeasured:
                value = (2.0 * math.sqrt(self.beta) * stds["flops"][idx]) / self.flops_cost
                if value > best_value:
                    best_choice = (point, "flops")
                    best_value = value

        if best_choice is None:
            return self._sample_new_point(), "both"
        return best_choice

    def _ask_with_original_flexibo(
        self,
        candidates: Sequence[Sequence[Any]],
        means: Dict[str, np.ndarray],
        stds: Dict[str, np.ndarray],
    ) -> tuple[List[Any], str] | None:
        if self.original_sampling is None or self.original_utils is None:
            return None
        if not candidates:
            return None

        try:
            beta_sqrt = math.sqrt(self.beta)
            region = []
            for idx in range(len(candidates)):
                # FlexiBO's reference implementation maximizes both axes when
                # constructing the pessimistic/optimistic region. EDGE objectives
                # are minimization targets, so use negative error/FLOPs utilities.
                error_mean = float(means["error"][idx])
                error_std = float(stds["error"][idx])
                flops_mean = float(means["flops"][idx])
                flops_std = float(stds["flops"][idx])
                region.append(
                    {
                        "pes": [
                            -(error_mean + beta_sqrt * error_std),
                            -(flops_mean + beta_sqrt * flops_std),
                        ],
                        "avg": [-error_mean, -flops_mean],
                        "opt": [
                            -(error_mean - beta_sqrt * error_std),
                            -(flops_mean - beta_sqrt * flops_std),
                        ],
                    }
                )

            undominated_indices, undominated = self.original_utils.identify_undominated_points(region)
            if not undominated:
                return None

            pess_pareto, pess_map = self.original_utils.construct_pessimistic_pareto_front(
                undominated_indices, undominated, "CONSTRUCT"
            )
            opt_pareto, opt_map = self.original_utils.construct_optimistic_pareto_front(
                undominated_indices, undominated, "CONSTRUCT"
            )
            if pess_map != opt_map:
                return None
            pess_volume = self.original_utils.compute_pareto_volume(pess_pareto)
            opt_volume = self.original_utils.compute_pareto_volume(opt_pareto)
            chosen_idx, chosen_point, chosen_objective = self.original_sampling.determine_next_sample(
                pess_pareto,
                opt_pareto,
                pess_map,
                opt_map,
                pess_volume,
                opt_volume,
                region,
                [list(point) for point in candidates],
            )
            objective = "accuracy" if chosen_objective == "o1" else "flops"
            point = list(chosen_point)
            obs = self.observations.get(point_key(point))
            if obs is not None:
                if objective == "accuracy" and obs.error is not None and obs.flops_norm is None:
                    objective = "flops"
                elif objective == "flops" and obs.flops_norm is not None and obs.error is None:
                    objective = "accuracy"
                elif obs.error is not None and obs.flops_norm is not None:
                    return None
            return point, objective
        except Exception as exc:
            print(f"FlexiBO original sampling fallback: {exc}")
            return None

    def _sample_new_point(self) -> List[Any]:
        for _ in range(1000):
            point = self.space.rvs(n_samples=1, random_state=self.rng)[0]
            if point_key(point) not in self.observations:
                return list(point)
        return list(self.space.rvs(n_samples=1, random_state=self.rng)[0])

    def partially_measured_accuracy_candidate(self) -> List[Any] | None:
        candidates = [
            obs.point
            for obs in self.observations.values()
            if obs.error is None and obs.flops_norm is not None
        ]
        if not candidates:
            return None
        return list(candidates[int(self.rng.randint(0, len(candidates)))])

    def _candidate_pool(self) -> List[List[Any]]:
        candidates: List[List[Any]] = []
        local_seen = set()

        incomplete = [
            obs.point
            for obs in self.observations.values()
            if obs.error is None or obs.flops_norm is None
        ]
        self.rng.shuffle(incomplete)
        for point in incomplete[: self.candidate_pool // 2]:
            key = point_key(point)
            local_seen.add(key)
            candidates.append(list(point))

        attempts = 0
        while len(candidates) < self.candidate_pool and attempts < self.candidate_pool * 100:
            attempts += 1
            point = self.space.rvs(n_samples=1, random_state=self.rng)[0]
            key = point_key(point)
            if key in local_seen:
                continue
            local_seen.add(key)
            candidates.append(list(point))

        return candidates or [self._sample_new_point()]

    def _predict_objectives(self, candidates: Sequence[Sequence[Any]]) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        x_candidates = np.asarray([self.encode(point) for point in candidates], dtype=np.float64)
        means: Dict[str, np.ndarray] = {}
        stds: Dict[str, np.ndarray] = {}

        for objective, attr in [("error", "error"), ("flops", "flops_norm")]:
            train = [
                (obs.point, getattr(obs, attr))
                for obs in self.observations.values()
                if getattr(obs, attr) is not None
            ]
            if len(train) < 2:
                fallback = 1.0
                if train:
                    fallback = float(train[0][1])
                means[objective] = np.full(len(candidates), fallback, dtype=np.float64)
                stds[objective] = np.ones(len(candidates), dtype=np.float64)
                continue

            x_train = np.asarray([self.encode(point) for point, _ in train], dtype=np.float64)
            y_train = np.asarray([float(y) for _, y in train], dtype=np.float64)
            means[objective], stds[objective] = self._predict_region(x_train, y_train, x_candidates)

        return means, stds

    def _optimistic_pareto_indices(
        self,
        means: Dict[str, np.ndarray],
        stds: Dict[str, np.ndarray],
    ) -> List[int]:
        opt_error = means["error"] - (math.sqrt(self.beta) * stds["error"])
        opt_flops = means["flops"] - (math.sqrt(self.beta) * stds["flops"])
        points = np.column_stack([opt_error, opt_flops])
        pareto = []
        for i, point in enumerate(points):
            dominated = False
            for j, other in enumerate(points):
                if i == j:
                    continue
                if np.all(other <= point) and np.any(other < point):
                    dominated = True
                    break
            if not dominated:
                pareto.append(i)
        return pareto

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
        if self.original_gp is not None:
            model, _ = self.original_gp.fit_gp()
            model.fit(x_train, y_train)
            mean, std = model.predict(x_candidates, return_std=True)
            return np.asarray(mean, dtype=np.float64).reshape(-1), np.asarray(std, dtype=np.float64).reshape(-1)

        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern, WhiteKernel
        except ModuleNotFoundError as exc:
            raise RuntimeError("scikit-learn is required for the FlexiBO GP surrogate.") from exc

        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-6)
        model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=self.seed + self.accuracy_evals,
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
        if self.original_rf is not None:
            model, _ = self.original_rf.fit_rf()
            model.set_params(random_state=self.seed + self.accuracy_evals, n_jobs=1)
            model.fit(x_train, y_train)
            tree_predictions = np.asarray(
                [tree.predict(x_candidates) for tree in model.estimators_],
                dtype=np.float64,
            )
            return tree_predictions.mean(axis=0).reshape(-1), tree_predictions.std(axis=0).reshape(-1)

        try:
            from sklearn.ensemble import RandomForestRegressor
        except ModuleNotFoundError as exc:
            raise RuntimeError("scikit-learn is required for the FlexiBO RF surrogate.") from exc

        model = RandomForestRegressor(
            n_estimators=64,
            min_samples_leaf=1,
            random_state=self.seed + self.accuracy_evals,
            n_jobs=1,
        )
        model.fit(x_train, y_train)
        tree_predictions = np.asarray(
            [tree.predict(x_candidates) for tree in model.estimators_],
            dtype=np.float64,
        )
        return tree_predictions.mean(axis=0), tree_predictions.std(axis=0)


HISTORY_FIELDS = [
    "event",
    "accuracy_eval",
    "objective",
    "error",
    "accuracy",
    "flops_norm",
    "flops",
    "params",
    "score",
    "elapsed_sec",
    "params_json",
]


def append_history(log_path: Path, row: Dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with log_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
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


def parse_optional_float(row: Dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in {"", "None", None}:
        return None
    return float(value)


def load_history(log_path: Path, space: Space, optimizer: FlexiboOptimizer) -> int:
    if not log_path.exists():
        return 0

    with log_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "objective" not in row:
                raise ValueError("Cannot resume old single-objective FlexiBO history. Use a new RESULTS_DIR.")
            point = point_from_params_json(space, row["params_json"])
            objective = row["objective"]
            result = {
                "error": parse_optional_float(row, "error"),
                "accuracy": parse_optional_float(row, "accuracy"),
                "flops_norm": parse_optional_float(row, "flops_norm"),
                "flops": parse_optional_float(row, "flops"),
                "params": parse_optional_float(row, "params"),
                "score": parse_optional_float(row, "score"),
            }
            optimizer.tell(point, objective, result)
    return optimizer.accuracy_evals


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    exp_dir = Path(args.name)
    overrides = vars(args).copy()
    overrides["opt"] = "RS"
    overrides["external_tuner"] = "FlexiBO"
    overrides["search_space_source"] = "RS"
    if "flops_module" not in overrides["mod_list"]:
        overrides["mod_list"] = list(overrides["mod_list"]) + ["flops_module"]

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
    flops_scale = float(args.flops_scale or args.flops_th or 1.0)
    evaluator = FlexiboEvaluator(base_space, ctrl, flops_scale=flops_scale)
    optimizer = FlexiboOptimizer(
        space=base_space,
        seed=cfg.seed,
        init_random=args.init_random,
        candidate_pool=args.candidate_pool,
        surrogate=args.surrogate,
        beta=args.beta,
        accuracy_cost=args.accuracy_cost,
        flops_cost=args.flops_cost,
    )

    print("\nSTARTING MULTI-OBJECTIVE FLEXIBO BASELINE\n")
    print("O1: validation error = 1 - accuracy")
    print(f"O2: normalized FLOPs = FLOPs / {flops_scale}")
    start_time = time.time()
    history_path = Path(cfg.name) / "algorithm_logs" / "flexibo_history.csv"

    completed_evals = 0
    flops_only_streak = 0
    if args.resume:
        completed_evals = load_history(history_path, base_space, optimizer)
        if completed_evals:
            print(f"[INFO] Resumed {completed_evals} previous accuracy evaluations from {history_path}.")
        else:
            print("[INFO] Resume requested, but no previous FlexiBO history was found. Starting fresh.")
        if history_path.exists():
            with history_path.open(newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("objective") == "flops":
                        flops_only_streak += 1
                    elif row.get("objective") in {"accuracy", "both"}:
                        flops_only_streak = 0

    target_eval = args.eval
    if args.max_new_evals is not None:
        if args.max_new_evals < 1:
            raise ValueError("--max_new_evals must be >= 1 when provided.")
        target_eval = min(args.eval, completed_evals + args.max_new_evals)

    event = sum(1 for _ in history_path.open()) - 1 if history_path.exists() else 0
    while optimizer.accuracy_evals < target_eval:
        if ctrl.convergence:
            print("[INFO] Controller early stopping triggered.")
            break

        event += 1
        point, objective = optimizer.ask()
        if objective == "flops" and flops_only_streak >= args.max_flops_only_streak:
            accuracy_point = optimizer.partially_measured_accuracy_candidate()
            if accuracy_point is not None:
                point = accuracy_point
                objective = "accuracy"
        print(f"\n--- FLEXIBO EVENT {event}; accuracy evals {optimizer.accuracy_evals} / {args.eval}; objective={objective} ---")

        if objective == "flops":
            result = evaluator.evaluate_flops_only(point)
        elif objective == "accuracy":
            result = evaluator.evaluate_accuracy(point)
        elif objective == "both":
            result = evaluator.evaluate_accuracy(point)
        else:
            raise ValueError(f"Unknown FlexiBO objective: {objective}")

        obs = optimizer.tell(point, objective, result)
        if objective == "flops":
            flops_only_streak += 1
        else:
            flops_only_streak = 0
        append_history(
            history_path,
            {
                "event": event,
                "accuracy_eval": optimizer.accuracy_evals,
                "objective": objective,
                "error": obs.error,
                "accuracy": obs.accuracy,
                "flops_norm": obs.flops_norm,
                "flops": obs.flops,
                "params": obs.params,
                "score": obs.score,
                "elapsed_sec": round(time.time() - start_time, 4),
                "params_json": json_params(base_space, point),
            },
        )

    total_time = time.time() - start_time
    print("\nMULTI-OBJECTIVE FLEXIBO BASELINE FINISHED")
    print(f"TOTAL TIME --------> {total_time:.2f} seconds")
    print(f"ACCURACY EVALS ----> {optimizer.accuracy_evals} / {args.eval}")
    print(f"HISTORY -----------> {history_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
