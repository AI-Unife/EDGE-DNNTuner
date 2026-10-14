from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional


class Sampler(ABC):
    """
    Search algorithm driven one evaluation at a time by the outer tuning loop.
    Each call evaluates `objective` once and returns an skopt-like result
    (with `x_iters` and `func_vals` covering every point seen so far).
    """

    @abstractmethod
    def step(self, objective, base_space, const_space, x0, y0,
             space_constraint: Optional[Callable] = None, callback=None):
        """
        Regular step.
          - base_space:  static reference space (expanded when rules add dimensions)
          - const_space: dynamic space as mutated by the symbolic rules
          - x0, y0:      points evaluated so far on the current space (empty = cold start)
        """

    @abstractmethod
    def fallback_step(self, objective, base_space,
                      space_constraint: Optional[Callable] = None, callback=None):
        """Recovery step used when `step` raises (e.g. GP matrix inversion failure)."""


@dataclass(frozen=True)
class OptStrategy:
    """
    Everything the tuner needs to know about an `opt` value.
      - sampler_factory: (seed, max_iter) -> Sampler
      - space_update:    how the search space evolves between iterations
                           "static"  -> never changes, no symbolic engine call
                           "expand"  -> rules mutate it, base space is expanded with new dims
                           "replace" -> rules mutate it, base space is replaced by the new one
      - use_filter:      pass the constraint filter to the sampler
    """
    sampler_factory: Callable[[int, int], Sampler]
    space_update: str
    use_filter: bool

    @property
    def use_rules(self) -> bool:
        return self.space_update != "static"

    @property
    def full_space(self) -> bool:
        """Without symbolic rules the space never grows, so it must expose every option up front."""
        return not self.use_rules
