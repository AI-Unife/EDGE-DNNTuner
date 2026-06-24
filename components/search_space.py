from __future__ import annotations

from typing import Dict, Any
import copy
from skopt.space import Integer, Real, Categorical, Space
from exp_config import load_cfg

class search_space:
    """
    Define and manage the hyperparameter search space used during iterative tuning.

    Notes:
    - Public API preserved (same class/method names & signatures).
    - We operate on `self.search_space` (an skopt Space) and manipulate dimensions by name.
    """

    def __init__(self) -> None:
        """
        Initialize helper epsilons used for building bounds of newly inserted dimensions.
        These are kept to preserve your original semantics.
        """
        self.epsilon_r1 = 10 ** -3
        self.epsilon_r2 = 10 ** 2
        self.epsilon_i = 2
        self.epsilon_d = 4
        
        self.cfg = load_cfg()

        # Will be set by search_sp(); defined here to be explicit.
        self.search_space: Space

    # ------------------------------------------------------------------ base --

    def search_sp(self, max_block=6, max_dense=4) -> Space:
        """
        Define the base search space reflecting the backbone of the neural network.
        Returns the created `skopt.space.Space`.
        """
        self.big_space = (self.cfg.opt in ['standard', 'RS']) or ("cim" in self.cfg.dataset)
        self.search_space = Space([
            Categorical(name='num_neurons', categories=[4, 8, 16, 32]),
            Integer(1, 4,  name='unit_c1'),
            Integer(1, 8, name='unit_c2'),
            Real(0.03, 0.5,  name='dr_f'),
            Real(1e-4, 1e-3, name='learning_rate'),
            Categorical(categories=[8, 16, 32, 64],  name='batch_size'),
            Categorical(['Adam', 'Adamax', 'SGD', 'Adagrad', 'Adadelta'], name='optimizer'),
            Categorical(['relu', 'elu', 'selu', 'swish'], name='activation'),
            Categorical(name='data_augmentation', categories=[False, True] if self.cfg.opt in ['standard', 'RS'] else [False]),
            Categorical(name="reg_l2", categories=[False, True] if self.cfg.opt in ['standard', 'RS'] else [False]),
            Categorical(name="skip_connection", categories=[False, True] if self.cfg.opt in ['standard', 'RS'] else [False])

        ])
        if self.big_space:
            for d in range(1,3):
                dense_name = f"fc_{d}"
                self.search_space.dimensions.append(Integer(0, 32, name=dense_name))
        for b in range(1, max_block + 1):
            conv_name = f'new_conv_{b}'
            if self.cfg.opt in ['standard', 'RS']:
                self.search_space.dimensions.append(Integer(0, 16, name=conv_name))
            else:
                self.search_space.dimensions.append(Integer(-1, 0, name=conv_name))
        for d in range(1, max_dense + 1):
            dense_name = f'new_fc_{d}'
            if self.cfg.opt in ['standard', 'RS']:
                self.search_space.dimensions.append(Integer(0, 32, name=dense_name))
            else:
                self.search_space.dimensions.append(Integer(-1, 0, name=dense_name))

        return self.search_space

    # --------------------------------------------------------- add / remove --

    def add_params(self, params: Dict[str, Any]) -> Space:
        """
        Reactivate existing "disabled" dimensions (low=high=0) in the search space
        using the value provided in `params` as the new `high` bound.

        Rules:
        - The dimension must ALREADY exist in the search space, otherwise a ValueError is raised.
        - A dimension is updated only if low=high=0 and params[name] > 0.
        - If the dimension is Integer -> replace with Integer(1, int(high_new))
            If the dimension is Real    -> replace with Real(1.0, float(high_new))
        - All other dimensions remain unchanged.
        """
        # Map name -> (index, dimension) for easy in-place replacement
        idx_by_name = {dim.name: (i, dim) for i, dim in enumerate(self.search_space.dimensions)}
        dims = list(self.search_space.dimensions)

        for name, value in params.items():
            if name not in idx_by_name:
                raise ValueError(f"Dimension '{name}' must already exist in the search space.")

            # Try to parse the passed value as numeric
            try:
                high_new_float = float(value)
            except Exception:
                # Non-numeric exemplar: skip silently to keep API stable
                continue

            if high_new_float <= 0:
                # Do not activate if the provided high value is not positive
                continue

            idx, dim = idx_by_name[name]
            low = getattr(dim, "low", None)
            high = getattr(dim, "high", None)

            # Activate only if the dimension is currently "off" (low=high=0)
            if low == -1 and high == 0:
                if isinstance(dim, Integer):
                    dims[idx] = Integer(-1, int(high_new_float), name=name)
                elif isinstance(dim, Real):
                    dims[idx] = Real(-1.0, float(high_new_float), name=name)
                else:
                    # Skip non-numeric dimensions (e.g., Categorical)
                    continue


        self.search_space = Space(dims)

        return self.search_space

    def expand_space(self, base_space, next_space):
        """
        Unisce next_space in base_space.
        - Se una dimensione esiste (stesso nome): allarga i bounds o le categorie.
        - Se una dimensione è nuova: la aggiunge alla fine.
        """
        
        # 1. Create a dictionary of current base_space dimensions for quick lookup
        # Key: dimension name, Value: dimension object
        base_dims_map = {d.name: d for d in base_space.dimensions}
        
        # List to track newly found dimension names
        new_dims_to_add = []

        # 2. Iterate over the dimensions of the proposed new space (next_space)
        for next_dim in next_space.dimensions:
            name = next_dim.name
            
            if name in base_dims_map:
                # --- CASE A: The dimension EXISTS -> UPDATE (Merge) ---
                base_dim = base_dims_map[name]
                
                if isinstance(next_dim, Categorical) and isinstance(base_dim, Categorical):
                        # Merge categories
                    current_cats = set(base_dim.categories)
                    new_cats = set(next_dim.categories)
                    if not new_cats.issubset(current_cats):
                        # Merge and convert back to tuple (keeping stable order if possible)
                        combined_cats = list(current_cats.union(new_cats))
                        new_categorical_dim = Categorical(
                            categories=combined_cats,
                            name=base_dim.name,
                            prior=None,          # <--- CRITICAL to avoid shape error
                        )
                        
                        # Replace the old object with the new one in the map
                        # So when we rebuild the final list, we'll take this updated one
                        base_dims_map[name] = new_categorical_dim
                        print(f"DEBUG: Extended categories for {name}: {new_categorical_dim}")

                elif isinstance(next_dim, (Integer, Real)) and isinstance(base_dim, (Integer, Real)):
                    # Widen the bounds (lowest minimum, highest maximum)
                    base_dim.low = min(base_dim.low, next_dim.low)
                    base_dim.high = max(base_dim.high, next_dim.high)
                    # print(f"DEBUG: Bounds expanded for {name}: {base_dim.low}, {base_dim.high}")
            
            else:
                # --- CASE B: The dimension DOES NOT EXIST -> NEW ---
                # Save it to add at the end
                new_dims_to_add.append(next_dim)

        # 3. Rebuild the ordered list of dimensions
        # NOTE: It is crucial to maintain the original base_space order to not break history (x0, y0)
        final_dimensions = [base_dims_map[d.name] for d in base_space.dimensions]
        
        # Append the new dimensions at the end
        for new_dim in new_dims_to_add:
            final_dimensions.append(new_dim)

        # 4. Return a NEW Space object
        # Important: recreating the Space object resets skopt's internal transformers
        return Space(final_dimensions)
    
    
    def remove_params(self, params: Dict[str, Any]) -> Space:
        """
        Deactivate existing dimensions in the search space when a value of 0 is provided.

        Behavior:
        - The dimension must ALREADY exist in the search space, otherwise a ValueError is raised.
        - If params[name] == 0, the dimension is replaced by:
            Integer(0, 0) if it was Integer
            Real(0.0, 0.0) if it was Real
        - All other dimensions remain unchanged.
        """
        # Map name -> (index, dimension) for in-place replacement
        idx_by_name = {dim.name: (i, dim) for i, dim in enumerate(self.search_space.dimensions)}
        dims = list(self.search_space.dimensions)

        for name, value in params.items():
            if name not in idx_by_name:
                raise ValueError(f"Dimension '{name}' must already exist in the search space.")

            # Skip if the provided value is not 0 (no deactivation needed)
            try:
                numeric_value = float(value)
            except Exception:
                # Ignore non-numeric values to keep API stable
                continue

            if numeric_value != 0:
                continue

            idx, dim = idx_by_name[name]

            # Replace the dimension with a "disabled" version
            if isinstance(dim, Integer):
                dims[idx] = Integer(-1, 0, name=name)
            elif isinstance(dim, Real):
                dims[idx] = Real(-1.0, 0.0, name=name)
            else:
                # Non-numeric dimensions (e.g., Categorical) are skipped
                continue

        self.search_space = Space(dims)

        return self.search_space

    # ----------------------------------------------------------- inspectors --

    def count_layer(self, type: str) -> int:
        """
        Count how many dimensions whose name contains `type`.
        E.g., `count_layer('new_conv')`.
        """
        return sum(1 for hp in self.search_space.dimensions if type in hp.name)

    # -------------------------------------------------------------- copies --

    def get_copy(self, space: Space, constrain_fn=None):
        """
        Return a **copy of the dimension list** from `space`.

        Notes:
            - Kept signature & return type to match your usage.
            - This returns the list of dimensions (not a Space), exactly like your original.
        """
        self.search_space = copy.deepcopy(space.dimensions)
        return self.search_space

    def reset_space(self, space: Space) -> Space:
        """
        Reset the current search space by ensuring all dimensions present in `space`
        are included here (union by name, preserving existing ones).

        Args:
            space: reference Space whose dimensions should be present.

        Returns:
            The updated `self.search_space` (an skopt Space).
        """
        current_names = {dim.name for dim in self.search_space.dimensions}
        new_dims = list(self.search_space.dimensions)  # start with current

        for dim in space.dimensions:
            if dim.name not in current_names:
                new_dims.append(copy.deepcopy(dim))

        self.search_space = Space(new_dims)
        return self.search_space


# ------------------------------ quick check ----------------------------------

if __name__ == '__main__':
    ss = search_space()
    sp = ss.search_sp()

    dtest = {'reg': 1e-4}
    res_final = ss.add_params(dtest)

    print(sp)
    print("-----------------------")
    print(res_final)