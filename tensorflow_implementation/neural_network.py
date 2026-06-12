import re
import numpy as np
from typing import Any, Dict, List, Tuple
from datetime import datetime
import random
import os

from components.neural_network import NeuralNetwork as BaseNeuralNetwork
from components.backend_interface import BackendInterface
from components.dataset import TunerDataset
from tensorflow_implementation.model import TFModel
from tensorflow_implementation.custom_train import train_model, eval_model

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Optimizer


# ----------------------------- Utilities -------------------------------------



def _optimizer_from_name(name: str, lr: float) -> Optimizer:
    """
    Build a Keras optimizer instance from a name string safely (no eval).
    Accepts canonical names like 'Adam', 'RMSprop', 'SGD' etc.
    """
    # tf.keras.optimizers.get accepts either a string or a config dict.
    try:
        # Prefer a config so we can set LR explicitly regardless of defaults.
        return tf.keras.optimizers.get({"class_name": name, "config": {"learning_rate": lr}})
    except Exception:
        # Fallback: try to resolve by attribute (case sensitive)
        opt_cls = getattr(tf.keras.optimizers, name, None)
        if opt_cls is None:
            raise ValueError(f"Unknown optimizer '{name}'.")
        return opt_cls(learning_rate=lr)


# ----------------------- Layer-wise LR wrapper -------------------------------

# class wrapper used to add the functionality of the layer wise learning rate
@keras.utils.register_keras_serializable()
class LayerWiseLR(Optimizer):
    def __init__(self, optimizer, multiplier, learning_rate=0.001, name="LWLR", **kwargs):
        """
        class wrapper used to add the functionality of the layer wise learning rate
        """
        # checks for the presence of the _HAS_AGGREGATE_GRAD attribute,
        # present since version 2.11 with the introduction of the new optimizer APIs,
        # to determine how to initialize the wrapper instance
        if hasattr(Optimizer, "_HAS_AGGREGATE_GRAD"):
            # wrapper initialization with new APIs, with learning rate stored in an internal slot
            super().__init__(name, **kwargs)
            self._set_hyper("learning_rate", learning_rate)
        else:
            # wrapper initialization with the old API, with the learning rate as an argument
            super().__init__(name=name, **kwargs)

        # storage of the attributes in the wrapper instance
        self._learning_rate = learning_rate #tf.Variable(learning_rate, trainable=False, dtype=tf.float32)
        self._optimizer = optimizer
        self._multiplier = multiplier

        self.last_grad_global_norm = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.last_grad_max_norm    = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.last_grad_mean_norm   = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    def apply_gradients(self, grads_and_vars, name: str = None, experimental_aggregate_gradients: bool = True):
        updated = []
        per_var_norms = []
        clean_grads = []

        for grad, var in grads_and_vars:
            if grad is not None:
                layer_name = (getattr(var, "path", None) or var.name).split("/")[0]
                scale = self._multiplier.get(layer_name, 1.0)
                g = grad * scale
                updated.append((g, var))
                clean_grads.append(g)
                per_var_norms.append(tf.norm(g))
            else:
                updated.append((grad, var))

        # Compute and save graph
        if clean_grads:
            gnorm = tf.linalg.global_norm(clean_grads)
            gmax  = tf.reduce_max(per_var_norms)
            gmean = tf.reduce_mean(per_var_norms)
            self.last_grad_global_norm.assign(gnorm)
            self.last_grad_max_norm.assign(gmax)
            self.last_grad_mean_norm.assign(gmean)
        else:
            self.last_grad_global_norm.assign(0.0)
            self.last_grad_max_norm.assign(0.0)
            self.last_grad_mean_norm.assign(0.0)

        self._optimizer.learning_rate.assign(self._learning_rate)

        return self._optimizer.apply_gradients(updated)

    # Delegate required slots/config to the base optimizer
    def _create_slots(self, var_list):
        if hasattr(self._optimizer, "_create_slots"):
            self._optimizer._create_slots(var_list)

    def get_config(self):
        # Parent config (includes clipnorm/clipvalue, etc.)
        cfg = super().get_config()

        # serialize the inner optimizer in a Keras-friendly way
        base_opt_cfg = keras.optimizers.serialize(self._optimizer)

        # multiplier must be JSON-serializable
        mult_cfg = {str(k): float(v) if hasattr(v, "__float__") else v
                    for k, v in self._multiplier.items()}

        cfg.update({
            "name": getattr(self, "name", self.__class__.__name__),
            "optimizer": base_opt_cfg,
            "multiplier": mult_cfg
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        # Extract what is needed for __init__
        name = config.pop("name", "LWLR")

        opt_cfg = config.pop("optimizer", None)
        optimizer = keras.optimizers.deserialize(opt_cfg) if isinstance(opt_cfg, dict) else opt_cfg
        mult = config.pop("multiplier", None)

        lr = config.get("learning_rate", None)

        return cls(optimizer=optimizer,
                   multiplier=mult,
                   learning_rate=lr,
                   name=name,
                   **config)



class NeuralNetwork(BaseNeuralNetwork):
    def __init__(self, backend:BackendInterface, dataset: TunerDataset, da: bool, reg: bool, residual: bool):
        super().__init__(backend, dataset, da, reg, residual)
        # Framework-specific preprocessing
        if self.dataset.n_classes > 1 and self.dataset.Y_train.ndim == 1:
            self.dataset.Y_train = tf.keras.utils.to_categorical(self.dataset.Y_train, self.dataset.n_classes)
            self.dataset.Y_test = tf.keras.utils.to_categorical(self.dataset.Y_test, self.dataset.n_classes)
        if self.dataset.X_train.ndim == 3:
            self.dataset.X_train = self.dataset.X_train[..., None]
            self.dataset.X_test = self.dataset.X_test[..., None]



    # --------------------------- Build the model -----------------------------


    def build_network(self, params, layer_x_block=2):
        """
        Define (or reload) the network architecture from params.

        Strategy:
          1) Try to resume from latest dashboard model if present.
          2) Otherwise build a new model based on `params`.
        """
        # 1) clear session
        tf.keras.backend.clear_session()
        if (self.exp_cfg.mode in ("fwdPass", "hybrid")) and "gesture" in self.exp_cfg.dataset :
            input_shape = self.dataset.X_train.shape[2:]  # (H, W, C) for gesture pipeline
            pos_input_shape = self.dataset.pos_train.shape[2:] if self.is_roi else None
        else:
            input_shape = self.dataset.X_train.shape[1:]  # generic (H, W, C)
            pos_input_shape = self.dataset.pos_train.shape[1:] if self.is_roi else None
        
        print(f"Building model with input_shape={input_shape}, pos_input_shape={pos_input_shape}, layer_x_block={layer_x_block}")
        # Model will handle separate branches
        self.model = TFModel(
            input_shape=input_shape,
            params=params,
            is_roi=self.is_roi,
            n_classes=self.dataset.n_classes,
            pos_input_shape=pos_input_shape,
            layer_x_block=layer_x_block
            )
        if self.exp_cfg.verbose > 1:
            self.model.model.summary()
        # if "flops_module" in self.exp_cfg.mod_list:
        # Compute FLOPs (approximate; counts MACs as 2 FLOPs)
        # Use same shape logic as build_network: shape[2:] for gesture fwdPass/hybrid data
        if (self.exp_cfg.mode in ("fwdPass", "hybrid")) and "gesture" in self.exp_cfg.dataset:
            data_shape = self.dataset.X_train.shape[2:]
            if self.is_roi:
                pos_shape = self.dataset.pos_train.shape[2:]
        else:
            data_shape = self.dataset.X_train.shape[1:]
            if self.is_roi:
                pos_shape = self.dataset.pos_train.shape[1:]

        if self.is_roi:
            # For ROI: pass both data and pos input shapes
            input_shapes = [data_shape, pos_shape]
        else:
            # For regular: pass only data input shape
            input_shapes = [data_shape]
        self.flops, self.nparams = self.backend.get_flops(self.model.model, input_shapes)
        if "hardware_module" in self.exp_cfg.mod_list:
            # Compute total latency cost
            from modules.loss.hardware_module import hardware_module
            HW_module = hardware_module(weight_cost=0.7)
            HW_module.update_state(self.model)
            self.tot_latency_cost = HW_module.total_cost


    def training(self, params: Dict[str, Any]) -> Tuple[List[float], Dict[str, List[float]], Model]:
        """
        Compile and train the model.

        Args:
            params: Hyperparameters (expects keys like unit_c1, unit_c2, unit_d, activation,
                    dr1_2, dr_f, optimizer (str), learning_rate (float), batch_size (int), [reg]).

        Returns:
            (score, history, model) where:
              - score: [loss, accuracy] from evaluation
              - history: Keras-like history dict
              - model: trained (and reloaded) Keras model
        """
        if self.model.model is None:
            print("Error: Model is not built.")
            exit(1)
        
        
        # Unique id for artifacts of this run
        model_name_id = datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")


        # Persist the (fresh) architecture JSON for later reloading
        model_json_path = f"{self.exp_cfg.name}/Model/model-{model_name_id}.json"
        with open(model_json_path, "w") as json_file:
            json_file.write(self.model.model.to_json())

        # Try loading previous weights if available (fine-tune / warm start)
        try:
            prev_weights = f"{self.exp_cfg.name}/Weights/weights.h5"
            if os.path.exists(prev_weights):
                self.model.model.load_weights(prev_weights)
        except Exception:
            pass  # ignore if incompatible

        # TensorBoard callback
        tb_log_dir = f"{self.exp_cfg.name}/log_folder/logs/{model_name_id}"
        tensorboard = TensorBoard(log_dir=tb_log_dir)

        # --- Optimizer (safe construction, then wrap with LayerWiseLR) ---
        base_opt = _optimizer_from_name(params["optimizer"], float(params["learning_rate"]))

        # Build per-layer multipliers (decay by sqrt(2) across Conv2D feature hierarchy)
        multiplier: Dict[str, float] = {}
        trainable_names = [
            (getattr(var, "path", None) or var.name).split("/")[0]
            for var in self.model.model.trainable_variables
        ]

        current_mul = 1.0
        lr_factor = 1.414213  # sqrt(2)
        for layer_name in trainable_names[::2]:  # skip bias variables (kernel/bias pairs)
            layer_type = self.model.model.get_layer(layer_name).__class__.__name__
            if layer_type in {"Conv2D"}:
                multiplier[layer_name] = current_mul
                current_mul /= lr_factor

        opt = LayerWiseLR(base_opt, multiplier, learning_rate=float(params["learning_rate"]))
        self.model.optimizer = opt

        # --- Callbacks ---
        es = EarlyStopping(monitor="val_loss", min_delta=0.005, patience=10, verbose=1,
                            mode="min", restore_best_weights=True)
        # --- Compile ---
        self.model.model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        
        if self.is_roi:
            X_train = [self.dataset.X_train, self.dataset.pos_train]
            X_test = [self.dataset.X_test, self.dataset.pos_test]
        else:
            X_train = self.dataset.X_train
            X_test = self.dataset.X_test

        # --- Train ---
        # Check if "debug" is in the experiment name
        if 'debug' in self.exp_cfg.name.lower():
            # Generate random history
            history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
            
            for epoch in range(self.epochs):
                # Generate random values
                train_loss = float(np.random.uniform(0.5, 2.0))
                val_loss = float(np.random.uniform(0.5, 2.0))
                train_acc = float(np.random.uniform(0.4, 0.95))
                val_acc = float(np.random.uniform(0.4, 0.95))
                
                # Update history
                history['loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['accuracy'].append(train_acc)
                history['val_accuracy'].append(val_acc)
                
                # Generate random batch and timing info
                num_batches = 17
                epoch_time_ms = int(np.random.uniform(150, 200))
                step_time_ms = epoch_time_ms // num_batches
                
                print(f"Epoch {epoch+1}/{self.exp_cfg.epochs}")
                print(f"{num_batches}/{num_batches} - 0s - loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f} - {epoch_time_ms}ms/epoch - {step_time_ms}ms/step")
            
            # Return random final scores
            final_loss = float(np.random.uniform(0.5, 2.0))
            final_acc = float(np.random.uniform(0.4, 0.95))
            score = [final_loss, final_acc]
            
        else: 
            if (self.exp_cfg.mode in ("fwdPass", "hybrid")) and "gesture" in self.exp_cfg.dataset :
            
                history = train_model(
                    self.model.model, opt,
                    X_train, self.dataset.Y_train,
                    X_test, self.dataset.Y_test,
                    self.epochs, params,
                    [tensorboard, es]
                )
            else:
                history = self.model.model.fit(
                    X_train, self.dataset.Y_train,
                    epochs=self.epochs,
                    batch_size=int(params["batch_size"]),
                    verbose=2,
                    validation_data=(X_test, self.dataset.Y_test),
                    callbacks=[tensorboard, es],
                ).history
            # --- Evaluate ---
            score = self.eval_model()

        try:
            dash_model_path = f"{self.exp_cfg.name}/dashboard/model/model.keras"
            self.model.model.save(dash_model_path)

            if not getattr(self.model.model, "built", False) or self.model.model.inputs is None:
                self.model.model = tf.keras.models.load_model(dash_model_path, custom_objects={"LayerWiseLR": LayerWiseLR})
        except:
            pass
        try:
            os.remove(model_json_path)
        except OSError:
            pass

        return score, history, self.model
    
    def eval_model(self):
        if self.is_roi:
            X_test = [self.dataset.X_test, self.dataset.pos_test]
        else:
            X_test = self.dataset.X_test
        if (self.exp_cfg.mode in ("fwdPass", "hybrid")) and "gesture" in self.exp_cfg.dataset:
            score = eval_model(self.model.model, X_test, self.dataset.Y_test)
        else:        
            score = self.model.model.evaluate(X_test, self.dataset.Y_test, verbose=2)
        return score
    
    def save_model(self):
        """Helper to handle safe model saving."""
        if self.model.model is None: return
        try:
            save_path = os.path.join(self.exp_cfg.name, "Model")
            os.makedirs(save_path, exist_ok=True)
            self.model.model.save(os.path.join(save_path, "best-model.keras"))
        except Exception as e:
            print(f"[ERROR] Failed to save best model: {e}")