from __future__ import annotations

from components.optimizers.base import Sampler


class BOSampler(Sampler):
    """Bayesian Optimization (GP + EI), one gp_minimize call per step, warm-started from x0/y0."""

    def __init__(self, gp_minimize, seed):
        self.gp_minimize = gp_minimize
        self.seed = seed

    def step(self, objective, base_space, const_space, x0, y0, space_constraint=None, callback=None):
        # If x0, y0 are empty, it's a "cold-start" (n_random_starts=1)
        # If x0, y0 are present, it's a "warm-start" (n_random_starts=0)
        is_warm_start = len(y0) > 0
        return self.gp_minimize(
            objective,
            base_space,
            x0=list(x0) if is_warm_start else None,
            y0=list(y0) if is_warm_start else None,
            acq_func="EI",
            n_calls=1,
            n_random_starts=0 if is_warm_start else 1,
            random_state=self.seed,
            callback=callback,
            space_constraint=space_constraint
        )

    def fallback_step(self, objective, base_space, space_constraint=None, callback=None):
        # Cold gp_minimize call with n_random_starts=1 to get a new point
        return self.gp_minimize(
            objective, base_space,
            acq_func="EI", n_calls=1, n_random_starts=1,
            random_state=self.seed, callback=callback,
            space_constraint=space_constraint
        )


def custom_bo(seed, max_iter) -> BOSampler:
    """BO with the project's gp_minimize (components/bo)."""
    from components.bo.gp import gp_minimize
    return BOSampler(gp_minimize, seed)


def skopt_bo(seed, max_iter) -> BOSampler:
    """BO with the stock scikit-optimize gp_minimize."""
    from skopt import gp_minimize
    return BOSampler(gp_minimize, seed)
