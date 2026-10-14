"""
Registry of the `opt` values accepted by the tuner.

Each name maps to an OptStrategy (sampler + space policy). To add a new
combination add one entry here; to add a new search algorithm implement a
Sampler in its own module and reference its factory below.
"""
from components.optimizers.base import OptStrategy, Sampler
from components.optimizers.bo import custom_bo, skopt_bo
from components.optimizers.random_search import random_search

STRATEGIES = {
    "standard": OptStrategy(custom_bo,     space_update="static",  use_filter=False),
    "filtered": OptStrategy(custom_bo,     space_update="expand",  use_filter=True),
    "basic":    OptStrategy(skopt_bo,      space_update="replace", use_filter=False),
    "RS":       OptStrategy(random_search, space_update="static",  use_filter=False),
    "RS_ruled": OptStrategy(random_search, space_update="expand",  use_filter=False),
}


def get_strategy(name: str) -> OptStrategy:
    if name not in STRATEGIES:
        raise ValueError(f"Unknown opt '{name}'. Choose from: {sorted(STRATEGIES)}")
    return STRATEGIES[name]
