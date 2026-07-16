import json
from random import random
import re
from time import time
from typing import List, Dict, Any

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import regularizers as reg
from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    BatchNormalization,
    Flatten,
    MaxPooling2D,
    Dropout,
    Input,
    Add
)
from keras.optimizers import *
from components.colors import colors
from components.neural_network import NeuralNetwork
from components.model_interface import LayerSpec, TunerModel, LayerTypes, InsertPosition, Params


class TFModel(TunerModel):

    from_type_map = {
        LayerTypes.InputLayer: layers.InputLayer,
        LayerTypes.Conv2D: layers.Conv2D,
        LayerTypes.MaxPooling2D: layers.MaxPooling2D,
        LayerTypes.GlobalAveragePooling2D: layers.GlobalAveragePooling2D,
        LayerTypes.Dropout: layers.Dropout,
        LayerTypes.Dense: layers.Dense,
        LayerTypes.BatchNormalization: layers.BatchNormalization,
        LayerTypes.Flatten: layers.Flatten,
        LayerTypes.Concatenate: layers.Concatenate,
        LayerTypes.Add: layers.Add,
        LayerTypes.ELU: "elu",
        LayerTypes.ReLU: "relu",
        LayerTypes.SeLU: "selu",
        LayerTypes.SiLU: "swish",
        LayerTypes.Softmax: "softmax",
        LayerTypes.Activation: layers.Activation
    }
    to_type_map = {v: k for k, v in from_type_map.items()}
    to_type_map[layers.Activation] = "activation"
    to_type_map["silu"] = LayerTypes.SiLU

    def __init__(self, input_shape, params, n_classes, is_roi=False, pos_input_shape=None, layer_x_block=2):
        super(TFModel, self).__init__()
        
        self.input_shape = input_shape
        self.params = params
        self.n_classes = n_classes
        self.is_roi = is_roi
        self.pos_input_shape = pos_input_shape
        self.residual = params.get("skip_connection", False)
        self.reg = params.get("reg_l2", False)
        self.da = params.get("data_augmentation", False)

        print(f"\n\n{self.cfg.dataset.lower()}\n\n")
        batch = True if "tiny" in self.cfg.dataset.lower() or "cim" in self.cfg.dataset.lower() else False  # self.reg if self.reg else None
        self.model = None
        # 2) Build a new CNN

        reg_layer = reg.l2() if self.reg else None

        inputs = Input(self.input_shape)
        if self.da:
            inputs = tf.keras.layers.RandomFlip("horizontal")(inputs)
            inputs = tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode="nearest")(inputs)

        x = Conv2D(params["unit_c1"] * params['num_neurons'], (3, 3), padding="same")(inputs)
        x = Activation(params["activation"])(x)
        x = BatchNormalization()(x) if batch else x
        for _ in range(1, layer_x_block-1):
            x = Conv2D(params["unit_c1"] * params['num_neurons'], (3, 3), padding="same")(x)
            x = Activation(params["activation"])(x)
            x = BatchNormalization()(x) if batch else x
        if self.residual:
            x = Conv2D(params["unit_c1"] * params['num_neurons'] , (3, 3), padding="same")(x)
            x = self._add_residual(inputs, x, params['unit_c1'] * params['num_neurons'], params['activation'], reg_layer)
        else:
            x = Conv2D(params["unit_c1"] * params['num_neurons'], (3, 3), padding="same", kernel_regularizer=reg_layer)(x)
            x = Activation(params["activation"])(x)
            x = BatchNormalization()(x) if batch else x
        x = Dropout(params["dr_f"])(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        shortcut = x
        for _ in range(layer_x_block-1):
            x = Conv2D(params["unit_c2"] * params['num_neurons'], (3, 3), padding="same", kernel_regularizer=reg_layer)(x)
            x = Activation(params["activation"])(x)
            x = BatchNormalization()(x) if batch else x
        if self.residual:
            x = Conv2D(params["unit_c2"] * params['num_neurons'], (3, 3), padding="same", kernel_regularizer=reg_layer)(x)
            x = self._add_residual(shortcut, x, params['unit_c2'] * params['num_neurons'], params['activation'], reg_layer)
        else:
            x = Conv2D(params["unit_c2"] * params['num_neurons'], (3, 3), padding="same", kernel_regularizer=reg_layer)(x)
            x = Activation(params["activation"])(x)
            x = BatchNormalization()(x) if batch else x
        x = MaxPooling2D(pool_size=(2, 2))(x)


        

        # Dynamically added conv blocks (conv -> act -> conv -> act -> pool -> dropout)
        added_convs = [k for k in params if re.match(r"new_conv_\d+$", k) and params[k] > 0]
        for layer_key in sorted(added_convs, key=lambda s: int(s.split("_")[-1])):  # stable order
            shortcut = x
            for _ in range(layer_x_block-1):
                x = Conv2D(params[layer_key] * params['num_neurons'], (3, 3), padding="same", kernel_regularizer=reg_layer)(x)
                x = Activation(params["activation"])(x)
                x = BatchNormalization()(x) if batch else x
            if self.residual:
                x = Conv2D(params[layer_key] * params['num_neurons'], (3, 3), padding="same", kernel_regularizer=reg_layer)(x)
                x = self._add_residual(shortcut, x, params[layer_key] * params['num_neurons'], params['activation'], reg_layer)
            else:
                x = Conv2D(params[layer_key] * params['num_neurons'], (3, 3), padding="same", kernel_regularizer=reg_layer)(x)
                x = Activation(params["activation"])(x)
                x = BatchNormalization()(x) if batch else x
            
            # x = Dropout(params["dr_f"])(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

        x = GlobalAveragePooling2D()(x) # if batch else Flatten()(x)
        
        # If ROI dataset, concatenate flattened pos with x
        pos_input = None
        # Now concatenate pos input if ROI dataset
        if self.is_roi:
            pos_input = Input(shape=self.pos_input_shape, name="pos_input")
            pos_flat = Flatten()(pos_input)
            x = tf.keras.layers.Concatenate()([x, pos_flat])
        
        added_fcs = [k for k in params if re.match(r"fc_\d+$", k) and params[k] > 0]
        for layer_key in sorted(added_fcs, key=lambda s: int(s.split("_")[-1])):  # stable order
            x = Dense(params[layer_key] * params['num_neurons'], kernel_regularizer=reg_layer)(x)
            x = Activation(params["activation"])(x)
            x = Dropout(params["dr_f"])(x)

        # Dynamically added FC layers
        added_fcs = [k for k in params if re.match(r"new_fc_\d+$", k) and params[k] > 0]
        for layer_key in sorted(added_fcs, key=lambda s: int(s.split("_")[-1])):  # stable order
            x = Dense(params[layer_key] * params['num_neurons'], kernel_regularizer=reg_layer)(x)
            x = Activation(params["activation"])(x)
            x = Dropout(params["dr_f"])(x)

        outputs = Dense(self.n_classes, kernel_regularizer=reg_layer, activation="softmax")(x)

        # Build model with appropriate inputs
        if self.is_roi and pos_input is not None:
            self.model = Model(inputs=[inputs, pos_input], outputs=outputs)
        else:
            self.model = Model(inputs=inputs, outputs=outputs)
        
        self.create_specs()
   
   
    def _add_residual(self, shortcut, x, output_channels, activation, reg_layer):
        # If input and output channel dimensions differ, align them via 1x1 conv
        if shortcut.shape[-1] != output_channels:
            shortcut = Conv2D(output_channels, (1, 1), padding="same", kernel_regularizer=reg_layer)(shortcut)
        # Add skip connection
        x = Add()([shortcut, x])
        # Apply activation after addition (ResNet-style)
    
        x = Activation(activation)(x)
        return x
    
    def create_specs(self):
        self.layers = {}
        for layer in self.model.layers:
            self.layers[layer.name] = self.to_spec(layer)

    def get_input_shape(self):
        # fix for keras >= 3.4, the input layer is saved
        # in lists of lists, instead of a single list
        if isinstance(self.model.input, list):
            return self.model.input[0]

        return self.model.input

    def from_type(self, layer_type: LayerTypes):
        return self.from_type_map[layer_type]

    def to_type(self, layer: Any):
        if isinstance(layer, layers.Layer):
            return self.to_type_map[layer.__class__]

        return self.to_type_map[layer]

    def to_spec(self, layer: layers.Layer):
        layer_type = self.to_type(layer.__class__)
        if layer_type == LayerTypes.Conv2D:
            return LayerSpec(
                name=layer.name,
                type=layer_type,
                module=layer,
                params={
                    Params.IN_CHANNELS: layer.input.shape[3],
                    Params.IN_HEIGHT: layer.input.shape[1],
                    Params.IN_WIDTH: layer.input.shape[2],
                    Params.OUT_CHANNELS: layer.output.shape[3],
                    Params.OUT_HEIGHT: layer.output.shape[1],
                    Params.OUT_WIDTH: layer.output.shape[2],
                    Params.KERNEL_SIZE: layer.kernel_size,
                    Params.STRIDE: layer.strides,
                    Params.PADDING: layer.padding,
                    Params.BIAS: layer.use_bias,
                }
            )
        elif layer_type == LayerTypes.Dense:
            return LayerSpec(
                name=layer.name,
                type=layer_type,
                module=layer,
                params={
                    Params.IN_FEATURES: layer.input.shape[1],
                    Params.OUT_FEATURES: layer.output.shape[1],
                }
            )
        elif layer_type == LayerTypes.MaxPooling2D:
            return LayerSpec(
                name=layer.name,
                type=layer_type,
                module=layer,
                params={
                    Params.KERNEL_SIZE: layer.pool_size,
                    Params.STRIDE: layer.strides,
                }
            )
        elif layer_type == LayerTypes.GlobalAveragePooling2D:
            return LayerSpec(
                name=layer.name,
                type=layer_type,
                module=layer
            )
        elif layer_type == LayerTypes.Dropout:
            return LayerSpec(
                name=layer.name,
                type=layer_type,
                module=layer,
                params={
                    Params.DROPOUT_RATE: layer.rate
                }
            )
        elif layer_type == LayerTypes.Flatten:
            return LayerSpec(
                name=layer.name,
                type=layer_type,
                module=layer
            )
        elif layer_type == "activation":
            activation_function = layer.get_config()["activation"]
            return LayerSpec(
                name=layer.name,
                type=LayerTypes.Activation,
                module=layer,
                is_activation=True,
                params={
                    "activation_function": activation_function
                }
            )
        elif layer_type == LayerTypes.InputLayer:
            return LayerSpec(
                name=layer.name,
                type=layer_type,
                module=layer,
                params={
                    "input_shape": layer.input_shape,
                }
            )
        elif layer_type == LayerTypes.BatchNormalization:
            return LayerSpec(
                name=layer.name,
                type=layer_type,
                module=layer
            )
        elif layer_type == LayerTypes.Concatenate:
            return LayerSpec(
                name=layer.name,
                type=layer_type,
                module=layer
            )
        elif layer_type == LayerTypes.Add:
            return LayerSpec(
                name=layer.name,
                type=layer_type,
                module=layer
            )
        else:
            raise Exception("Missing LayerSpec for layer of type " + layer.__class__.__name__)

    def from_spec(self, layer_spec: LayerSpec):
        if layer_spec.type == LayerTypes.Conv2D:
            return layers.Conv2D(
                name=layer_spec.name,
                filters=layer_spec.get(Params.OUT_CHANNELS),
                kernel_size=layer_spec.get(Params.KERNEL_SIZE),
                padding="same" if layer_spec.get(Params.PADDING) == 1 else "valid",
                use_bias=layer_spec.get(Params.BIAS)
            )
        elif layer_spec.type == LayerTypes.Dense:
            return layers.Dense(
                name=layer_spec.name,
                units=layer_spec.get(Params.OUT_FEATURES)
            )
        elif layer_spec.type == LayerTypes.MaxPooling2D:
            return layers.MaxPooling2D(
                name=layer_spec.name,
                pool_size=layer_spec.get(Params.KERNEL_SIZE),
            )
        elif layer_spec.type == LayerTypes.GlobalAveragePooling2D:
            return layers.GlobalAveragePooling2D(
                name=layer_spec.name
            )
        elif layer_spec.type == LayerTypes.Dropout:
            return layers.Dropout(
                name=layer_spec.name,
                rate=layer_spec.get(Params.DROPOUT_RATE)
            )
        elif layer_spec.type == LayerTypes.Flatten:
            return layers.Flatten(
                name=layer_spec.name
            )
        elif layer_spec.type == LayerTypes.Activation: # [LayerTypes.ELU, LayerTypes.Softmax, LayerTypes.ReLU, LayerTypes.Sigmoid, LayerTypes.SiLU, LayerTypes.SeLU]:
            return layers.Activation(
                name=layer_spec.name,
                activation=layer_spec.get("activation_function"),
            )
        elif layer_spec.type == LayerTypes.InputLayer:
            return layers.InputLayer(
                name=layer_spec.name,
                input_shape=layer_spec.get("input_shape")[1:],
                batch_size=1
            )
        elif layer_spec.type == LayerTypes.BatchNormalization:
            return layers.BatchNormalization(
                name=layer_spec.name,
            )
        elif layer_spec.type == LayerTypes.Concatenate:
            return layers.Concatenate(
                name=layer_spec.name,
            )
        elif layer_spec.type == LayerTypes.Add:
            return layers.Add(
                name=layer_spec.name,
            )
        else:
            print("===================== FALLBACK ====================")
            print("TYPE:", layer_spec.type)
            return self.from_type(layer_spec.type)()
