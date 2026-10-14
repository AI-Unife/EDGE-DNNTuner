from __future__ import annotations

from components.optimizers.base import Sampler


class RSSampler(Sampler):
    """Random search: draws the next point of a seeded sequence from the current space."""

    def __init__(self, seed, max_iter):
        from components.random_search import RandomSearch
        self.rs = RandomSearch(random_state=seed, total_iter=max_iter)

    def step(self, objective, base_space, const_space, x0, y0, space_constraint=None, callback=None):
        return self.rs(objective, const_space, callback=callback)

    def fallback_step(self, objective, base_space, space_constraint=None, callback=None):
        return self.rs(objective, base_space, callback=callback)


def random_search(seed, max_iter) -> RSSampler:
    return RSSampler(seed, max_iter)
