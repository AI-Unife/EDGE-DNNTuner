from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple
from skopt.space import Categorical
import logging

try:
    # Pretty terminal colors (optional). Falls back gracefully if unavailable.
    from components.colors import colors
except Exception:  # pragma: no cover
    class _NoColor:
        FAIL = ""
        ENDC = ""
    colors = _NoColor()  # type: ignore


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class tuning_rules_symbolic:
    """
    Rule-based helper to tweak a model architecture and its hyperparameter search space.

    This class centralizes small, composable "repair" actions (add layers, widen search
    bounds, enable regularization, etc.) that can be invoked by a higher-level
    symbolic/diagnostic system when specific issues are detected during training.
    """

    # ----------------------------- Lifecycle ---------------------------------

    def __init__(self, params, ss, controller) -> None:
        """
        Initialize tuning context and bookkeeping.

        Args:
            params: The hyperparameter search space container used by `ss`.
            ss:     A search-space manager exposing `add_params`, `remove_params`,
                    and `count_initial_layers(...)`.
            controller: Object that owns the current model and exposes mutators like
                    `add_fc_layer`, `add_conv_section`, `remove_*`, `set_case`,
                    `set_data_augmentation`, and `manage_configuration`. It must also
                    provide training data shape via `controller.X_train.shape`.
        """
        self.space = params
        self.ss = ss
        self.controller = controller
        self.count_no_probs = 0

    # --------------------------- Regularization ------------------------------

    def reg_l2(self) -> None:
        """
        Enable L2 regularization once.
        """
        for i, hp in enumerate(self.space):
            if hp.name == "reg_l2":
                new_categories = [True]
                self.space.dimensions[i] = Categorical(new_categories, name='reg_l2')
        logger.info("Enabled L2 regularization")

    def remove_reg_l2(self) -> None:
        """
        Disable reg in the controller.
        """
        for i, hp in enumerate(self.space):
            if hp.name == "reg_l2":
                new_categories = [False]
                self.space.dimensions[i] = Categorical(new_categories, name='reg_l2')
        logger.info("Disable L2 regularization")

    def data_augmentation(self) -> None:
        """
        Enable data augmentation in the controller.
        """
        for i, hp in enumerate(self.space):
            if hp.name == "data_augmentation":
                new_categories = [True]
                self.space.dimensions[i] = Categorical(new_categories, name='data_augmentation')
        logger.info("Enabled data augmentation")

    def remove_data_augmentation(self) -> None:
        """
        Enable data augmentation in the controller.
        """
        for i, hp in enumerate(self.space):
            if hp.name == "data_augmentation":
                new_categories = [False]
                self.space.dimensions[i] = Categorical(new_categories, name='data_augmentation')
        logger.info("Disabled data augmentation")

    def add_residual(self) -> None:
        """
        Enable residual connection in the controller.
        """
        for i, hp in enumerate(self.space):
            if hp.name == "skip_connection":
                new_categories = [True]
                self.space.dimensions[i] = Categorical(new_categories, name='skip_connection')
        logger.info("Enabled residual connection")
        
    def remove_residual(self) -> None:
        """
        Disable residual connection in the controller.
        """
        for i, hp in enumerate(self.space):
            if hp.name == "skip_connection":
                new_categories = [False]
                self.space.dimensions[i] = Categorical(new_categories, name='skip_connection')
        logger.info("Disabled residual connection")
    # -------------------------- Architecture edits ---------------------------

    def new_fc_layers(self) -> None:
        """
        Add a fully-connected (dense) layer, respecting a soft upper bound to avoid bloat.
        """
        if self.controller.count_new_fc >= self.controller.max_fc:
            print(
                colors.FAIL,
                "Max number of dense layers reached",
                colors.ENDC,
            )
            print(
                colors.FAIL,
                " Dense layers: ",
                self.controller.count_new_fc,
                " Max dense layers: ",
                self.controller.max_fc,
                colors.ENDC,
            )
            return

        self.controller.count_new_fc += 1
        new_p = {f"new_fc_{self.controller.count_new_fc}": 32}
        self.space = self.ss.add_params(new_p)
        logger.info("Added dense layer #%d", self.controller.count_new_fc)

    def dec_fc_layers(self) -> None:
        """
        Remove one dense layer if present.
        """
        print("Removing dense layer")
        print(self.controller.count_new_fc)
        if self.controller.count_new_fc < 1:
            print(colors.FAIL, "No more dense layers to remove", colors.ENDC)
            self.controller.count_new_fc = 0
            return

        new_p = {f"new_fc_{self.controller.count_new_fc}": 0}
        self.space = self.ss.remove_params(new_p)
        logger.info("Removed one dense layer; remaining added layers: %d", self.controller.count_new_fc)
        self.controller.count_new_fc -= 1

    def new_conv_block(self) -> None:
        """
        Add a convolutional *section* (e.g., conv block). Respect input-size constraints.
        """
        if not self.controller.max_conv:
            return

        tot_conv = self.controller.count_new_cv
        if tot_conv >= self.controller.max_conv:
            print(colors.FAIL, "Max number of convolutional layers reached", colors.ENDC)
            print(
                colors.FAIL,
                "Convolutional layers: ",
                tot_conv - 2,
                " Max convolutional layers: ",
                self.controller.max_conv,
                colors.ENDC,
            )
            return

        self.controller.count_new_cv += 1
        new_p = {f"new_conv_{self.controller.count_new_cv}": 16}
        self.space = self.ss.add_params(new_p)
        logger.info("Added conv section #%d", self.controller.count_new_cv)

    def dec_conv_block(self) -> None:
        """
        Remove one convolutional section if present.
        """
        if self.controller.count_new_cv < 1:
            logger.info("No more convolutional section to remove")
            self.count_new_cv = 0
            return

        # Remove param key associated with the *next* (now absent) conv
        new_p = {f"new_conv_{self.controller.count_new_cv}": 0}
        self.space = self.ss.remove_params(new_p)
        logger.info("Removed one conv section; remaining added sections: %d", self.controller.count_new_cv)
        self.controller.count_new_cv -= 1

    def inc_conv_layers(self) -> None:
        if self.controller.layer_x_block < self.controller.max_layer_x_block:
            self.controller.layer_x_block += 1
            logger.info("Add one conv layer per section; new layers per sections: %d", self.controller.layer_x_block)
        logger.info("No more convolutional layers per section to add")

    def dec_conv_layers(self) -> None:
        if self.controller.layer_x_block < self.controller.max_layer_x_block:
            self.controller.layer_x_block -= 1
            logger.info("Removed one conv layer per section; remaining layers per sections: %d", self.controller.layer_x_block)
        logger.info("No more convolutional layers per section to remove")

    # -------------------------- Hyperparam tweaks ----------------------------

    def decr_lr(self, params):
        """
        method used to decrement learning rate
        """
        for hp in self.space:
            # check if learning rate is present in the search space and in that case
            # proceed by increasing the upper range adding half of its value
            if hp.name == 'learning_rate':
                hp.high = params['learning_rate'] + (params['learning_rate'] / 2)

    def inc_lr(self, params):
        """
        method used to increment learning rate
        """
        for hp in self.space:
            # check if learning rate is present in the search space and in that case
            # proceed by incresing the upper range with the current learning rate value
            if hp.name == 'learning_rate':
                hp.high = params['learning_rate'] + hp.high

    def inc_neurons(self, params):
        """
        method used to increment the number of neurons
        """
        # itereate over each hyperparameter and if one of these is a convolutional
        # or dense layer decrease the lower value of the range
        valid_names = ['unit_c1', 'unit_c2', 'unit_d', 'new_conv', 'new_fc', 'unit_p1']
        for hp in self.space:
            for name in valid_names:
                if name in hp.name and hp.high > 0:
                    hp.low = max(params[hp.name] - 1, 1)

    
    def inc_batch_size(self, params):
        """
        method used to increment batch_size
        """
        for i, hp in enumerate(self.space):
            if hp.name == 'batch_size':
                current_val = params.get('batch_size')
                if current_val is None:
                    raise ValueError("Parameter 'batch_size' not found in params.")
                if not isinstance(hp, Categorical):
                    raise TypeError("The 'batch_size' dimension must be Categorical.")

                # Keep only categories >= current batch size
                new_categories = [c for c in hp.categories if c >= current_val]

                if not new_categories:
                    raise ValueError(f"No valid categories >= {current_val} for 'batch_size'.")

                # Replace the dimension with the filtered one
                self.space.dimensions[i] = Categorical(new_categories, name='batch_size')

    def inc_dropout(self, params):
        """
        method used to increment dropout
        """

        # iterate over hyperparameters space
        for hp in self.space:
            # check if dropout is present in the search space and in that case
            # proceed by increasing the lower range
            if 'dr' in hp.name:
                hp.low = min(params[hp.name] - params[hp.name] / 100, 0.0)

    def dec_neurons(self, params):
        """
        method used to decrement the number of neurons
        """
        # itereate over each hyperparameter and if one of these is a convolutional
        # or dense layer increase the upper value of the range
        for hp in self.space:
            if 'unit_c1' in hp.name and hp.high > 0:
                hp.high = min(params['unit_c1'] + 1, 4)
            if 'unit_c2' in hp.name  and hp.high > 0:
                hp.high = min(params['unit_c2'] + 1, 8)
            if 'unit_p1' in hp.name and hp.high > 0:
                hp.high = min(params['unit_p1'] + 1, 4)
            if 'new_conv' in hp.name and hp.high > 0:
                try:
                    hp.high = min(params[hp.name] + 1, 16)
                except KeyError:
                    continue
            if 'new_fc' in hp.name and hp.high > 0:
                try:
                    hp.high = min(params[hp.name] + 1, 32)
                except KeyError:
                    continue

    def dec_batch_size(self, params):
        """
        method used to decrement batch_size
        """
        for i, hp in enumerate(self.space):
            if hp.name == 'batch_size':
                current_val = params.get('batch_size')
                if current_val is None:
                    raise ValueError("Parameter 'batch_size' not found in params.")
                if not isinstance(hp, Categorical):
                    raise TypeError("The 'batch_size' dimension must be Categorical.")

                # Keep only categories >= current batch size
                new_categories = [c for c in hp.categories if c <= current_val]

                if not new_categories:
                    raise ValueError(f"No valid categories <= {current_val} for 'batch_size'.")

                # Replace the dimension with the filtered one
                self.space.dimensions[i] = Categorical(new_categories, name='batch_size')

    def dec_dropout(self, params):
        """
        method used to decrement dropout
        """
        for hp in self.space:
            if 'dr' in hp.name:
                hp.high = max(params[hp.name] + params[hp.name] / 100, 0.8)

    # ------------------------- System configuration --------------------------

    def new_config(self) -> None:
        """
        Ask the controller to try a different hardware/run configuration.
        """
        self.controller.manage_configuration()
        logger.info("Requested a new hardware/run configuration")

    # ------------------------------- Orchestration ---------------------------

    def repair(self, sym_tuning: Iterable[str], diagnosis: Iterable[str], params: Dict[str, Any], const_space) -> Tuple[Any, Any]:
        """
        Execute a sequence of symbolic tuning actions produced by a diagnosis step.

        Args:
            sym_tuning: Iterable of method names (strings) to invoke on this instance.
                        Methods that require `params` should be listed without arguments;
                        this function detects whether to pass `params` or not based on
                        the method name (matching the original behavior).
            diagnosis:  Iterable of human-readable diagnoses parallel to `sym_tuning`.
            model:      The current model object (will be deleted from controller to force rebuild).
            params:     Dict of current parameter values to guide the tuning actions.

        Returns:
            Tuple of (updated_search_space, model). Note that `model` is returned
            unchanged here—the rebuild is assumed to be handled downstream.
        """
        # Drop the existing model object from the controller to avoid stale state.
        if hasattr(self.controller, "model"):
            del self.controller.model

        self.space = const_space
        self.ss.search_space = self.space
        diag_list = list(diagnosis)
        for i, action_name in enumerate(sym_tuning):
            # Preserve original control flow semantics:
            # - Some methods are called with params, others without.
            needs_params = (
                action_name in {
                    "decr_lr",
                    "inc_lr",
                    "inc_dropout",
                    "inc_neurons",
                    "inc_batch_size",
                    "dec_neurons",
                    "dec_batch_size",
                    "dec_dropout",
                }
            )

            if "X" in action_name:
                print("I have not found problem, but I'll try some new configurations anyway.")
                self.count_no_probs += 1
                break

            # Resolve the method safely (avoid eval)
            method = getattr(self, action_name, None)
            if not callable(method):
                logger.warning("Skipping unknown tuning action: %s", action_name)
                continue

            self.count_no_probs = 0
            readable_diag = diag_list[i] if i < len(diag_list) else "unspecified issue"
            verb = f"self.{action_name}({'params' if needs_params else ''})"
            print(f"I've find {readable_diag} and I'm trying to fix it with {verb}.")  # Keep original print UX

            # Invoke with or without params according to original rule
            if needs_params:
                method(params)
            else:
                method()

        return self.space
        