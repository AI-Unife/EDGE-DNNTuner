import sys
from typing import List
from skopt.space import Space, Categorical, Integer, Real

class ConstraintsWrapper:
    def __init__(self, space: Space):
        self.space = space

    def apply_constraints(self, params: List) -> bool:
        """
        Check whether a sampled parameter vector fits within the (possibly updated) space.
        """
        for i, dim in enumerate(self.space.dimensions):
            val = params[i]
            if isinstance(dim, Categorical):
                if val not in dim.categories:
                    return False
            elif isinstance(dim, (Integer, Real)):
                if not (dim.low <= val <= dim.high):
                    return False
            else:
                # Raise an error that the main program can catch if needed.
                raise TypeError(f"Type space dimension {dim} - {type(dim)} not valid")
        return True
