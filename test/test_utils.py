from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Optimizer


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

@keras.utils.register_keras_serializable()
class LayerWiseLR(Optimizer):
    """
    Compatibility optimizer wrapper used only to deserialize models
    that were saved with this custom optimizer.
    """

    def __init__(self, optimizer, multiplier, learning_rate=0.001, name="LWLR", **kwargs):
        if hasattr(Optimizer, "_HAS_AGGREGATE_GRAD"):
            super().__init__(name=name, **kwargs)
            self._set_hyper("learning_rate", learning_rate)
        else:
            super().__init__(name=name, **kwargs)

        self._learning_rate = learning_rate
        self._optimizer = optimizer
        self._multiplier = multiplier

    def apply_gradients(self, grads_and_vars, name: str = None, experimental_aggregate_gradients: bool = True):
        updated = []
        for grad, var in grads_and_vars:
            if grad is not None:
                layer_name = (getattr(var, "path", None) or var.name).split("/")[0]
                scale = self._multiplier.get(layer_name, 1.0)
                updated.append((grad * scale, var))
            else:
                updated.append((grad, var))
        self._optimizer.learning_rate.assign(self._learning_rate)
        return self._optimizer.apply_gradients(updated)

    def _create_slots(self, var_list):
        if hasattr(self._optimizer, "_create_slots"):
            self._optimizer._create_slots(var_list)

    def get_config(self):
        cfg = super().get_config()
        base_opt_cfg = keras.optimizers.serialize(self._optimizer)
        mult_cfg = {
            str(k): float(v) if hasattr(v, "__float__") else v
            for k, v in self._multiplier.items()
        }
        cfg.update({
            "name": getattr(self, "name", self.__class__.__name__),
            "optimizer": base_opt_cfg,
            "multiplier": mult_cfg,
            "learning_rate": self._learning_rate,
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        name = config.pop("name", "LWLR")
        opt_cfg = config.pop("optimizer", None)
        optimizer = keras.optimizers.deserialize(opt_cfg) if isinstance(opt_cfg, dict) else opt_cfg
        mult = config.pop("multiplier", None)
        lr = config.get("learning_rate", None)
        return cls(optimizer=optimizer, multiplier=mult, learning_rate=lr, name=name, **config)


def load_keras_model(model_path: str) -> tf.keras.Model:
    """Load model from <model_path>"""
    keras_path = Path(model_path)
    if not keras_path.exists():
        raise FileNotFoundError(f"Model not found at: {keras_path}")

    model = tf.keras.models.load_model(
        str(keras_path),
        custom_objects={"LayerWiseLR": LayerWiseLR},
        compile=False,
    )
    
    
    return model
