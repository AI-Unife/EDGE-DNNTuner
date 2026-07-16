import re
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from components.model_interface import InsertPosition, LayerSpec, TunerModel, LayerTypes, Params


class ConvBlock(nn.Module):
    """
    Gestisce un blocco di 'layer_x_block' convoluzioni.
    Se use_residual=True, aggiunge l'input all'output (ResNet style).
    """
    def __init__(self, in_channels, out_channels, num_repeats, activation_fn, use_residual, batch=True):
        super(ConvBlock, self).__init__()
        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        self.activation = activation_fn
        self.batch = batch
        
        # 1. Build the first N-1 layers — order matches TF: Conv -> Act -> BN
        current_in = in_channels
        
        # Add (num_repeats - 1) standard blocks
        for _ in range(num_repeats - 1):
            self.layers.append(nn.Sequential(
                nn.Conv2d(current_in, out_channels, kernel_size=3, padding=1, bias=False),
                copy.deepcopy(activation_fn),
                nn.BatchNorm2d(out_channels) if batch else nn.Identity(),
            ))
            current_in = out_channels  # After the first one, input is out_channels

        # 2. The last layer of the block (before the residual sum)
        self.last_conv = nn.Conv2d(current_in, out_channels, kernel_size=3, padding=1, bias=False)
        self.last_bn = nn.BatchNorm2d(out_channels)

        # 3. Skip Connection (Shortcut) handling
        # If input channels differ from output channels (e.g., transition from block 1 to 2),
        # a 1x1 conv is needed to adapt dimensions (Projection Shortcut).
        if use_residual:
            self.shortcut = nn.Identity()
            if (in_channels != out_channels):
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

    def forward(self, x):
        out = x

        # Forward pass through the first N-1 layers (Conv -> Act -> BN)
        for layer in self.layers:
            out = layer(out)

        # Last convolutional layer
        out = self.last_conv(out)

        if self.use_residual:
            # TF _add_residual: Conv -> Add(shortcut, x) -> Act  (no BN after Add)
            out = out + self.shortcut(x)
            out = self.activation(out)
        else:
            # TF non-residual: Conv -> Act -> [BN]
            out = self.activation(out)
            if self.batch:
                out = self.last_bn(out)

        return out


class TorchModel(nn.Module, TunerModel):
    """
    PyTorch implementation of a TunerModel that extends nn.Module.
    Provides bidirectional mapping between layer types and PyTorch modules,
    along with utilities for converting between layer specifications and actual modules.
    """

    # Maps custom layer types to PyTorch neural network modules
    from_type_map = {
        LayerTypes.Conv2D: nn.Conv2d,
        LayerTypes.MaxPooling2D: nn.MaxPool2d,
        LayerTypes.Dropout: nn.Dropout,
        LayerTypes.Dense: nn.Linear,
        LayerTypes.ELU: nn.ELU,
        LayerTypes.ReLU: nn.ReLU,
        LayerTypes.SiLU: nn.SiLU,
        LayerTypes.SeLU: nn.SELU,
        LayerTypes.Sigmoid: nn.Sigmoid,
        LayerTypes.BatchNormalization1D: nn.BatchNorm1d,
        LayerTypes.BatchNormalization2D: nn.BatchNorm2d,
        LayerTypes.Flatten: nn.Flatten
    }
    # Reverse mapping from PyTorch modules back to custom layer types
    to_type_map = {v: k for k, v in from_type_map.items()}

    def __init__(self, input_shape, params, n_classes, layer_x_block=2, batch=True,
                 is_roi=False, pos_input_shape=None):
        """
        Initialize the model with the given configuration.
        
        Args:
            input_shape:     (C, H, W) per-frame input dimensions.
            params:          Hyperparameter dict.
            n_classes:       Number of output classes.
            layer_x_block:   Convolutions per block (min 2 to match TF behaviour).
            batch:           Enable BatchNorm (True only for tiny-imagenet / cim).
            is_roi:          Whether the dataset provides a position map input.
            pos_input_shape: Shape of the position map (e.g. (4, 16, 16)); ignored
                             when is_roi=False.
        """
        super(TorchModel, self).__init__()
        
        self.input_shape = input_shape
        self.params = params
        self.n_classes = n_classes
        self.activation = self._get_activation(params['activation'])
        self.batch = batch
        self.layer_x_block = layer_x_block
        self.is_roi = is_roi
        self.pos_input_shape = pos_input_shape
        
            # Parameter parsing
        self.use_residual = params.get('skip_connection', False)
        # Activation instance (e.g., ReLU) to reuse
        act_fn = self._get_activation(params['activation'])
        self.batch = batch
        
        # Compute number of channels
        # unit_c * num_neurons
        c1_channels = int(params['unit_c1'] * params['num_neurons'])
        c2_channels = int(params['unit_c2'] * params['num_neurons'])
        
        # --- NETWORK CONSTRUCTION ---
        self.features = nn.Sequential()
        in_c = input_shape[0] # (Channels, H, W)
        
        # TF always executes a mandatory first conv before the loop, so the
        # minimum number of convolutions per block is 2 regardless of layer_x_block.
        # In TF only C1 has an unconditional "first conv" outside the loop, so:
        #   C1            → max(2, layer_x_block) total convolutions
        #   C2 + added    → layer_x_block total convolutions  (min 1)
        _c1_repeats    = max(2, layer_x_block)
        _other_repeats = max(1, layer_x_block)  # C2 and new_conv_* blocks

        # 1. Blocco C1
        self.features.add_module("block_c1", ConvBlock(
            in_channels=in_c, 
            out_channels=c1_channels, 
            num_repeats=_c1_repeats, 
            activation_fn=act_fn, 
            use_residual=self.use_residual,
            batch=self.batch
        ))
        self.features.add_module("pool1", nn.MaxPool2d(2))
        
        # 2. Blocco C2
        self.features.add_module("block_c2", ConvBlock(
            in_channels=c1_channels, 
            out_channels=c2_channels, 
            num_repeats=_other_repeats, 
            activation_fn=act_fn, 
            use_residual=self.use_residual,
            batch=self.batch
        ))
        self.features.add_module("pool2", nn.MaxPool2d(2))
        in_channels = c2_channels
        added_convs = [k for k in params if re.match(r"new_conv_\d+$", k) and params[k] > 0]
        for layer_key in sorted(added_convs, key=lambda s: int(s.split("_")[-1])):
            val = params[layer_key]
            out_c = int(val * params['num_neurons'])
            self.features.add_module(f"added_conv_{layer_key}", ConvBlock(
                in_channels=in_channels,
                out_channels=out_c,
                num_repeats=_other_repeats,
                activation_fn=act_fn,
                use_residual=self.use_residual,
                batch=self.batch
            ))
            self.features.add_module(f"pool_{layer_key}", nn.MaxPool2d(2))
            in_channels = out_c

        # --- CLASSIFICATORE (Fully Connected) ---
        # Compute flatten dimension from a dummy forward pass through features.
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out_feat = self.features(dummy)
            if self.batch:
                out_feat = F.adaptive_avg_pool2d(out_feat, (1, 1))
            self.flat_dim = out_feat.view(1, -1).size(1)

        # For ROI datasets, flatten and concatenate the position map exactly as TF:
        #   Flatten(features) + Flatten(pos_map) -> Concatenate -> Dense
        if self.is_roi and pos_input_shape is not None:
            self.pos_flat_dim = 1
            for d in pos_input_shape:
                self.pos_flat_dim *= d
        else:
            self.pos_flat_dim = 0

        self.classifier = nn.Sequential()
        current_dim = self.flat_dim + self.pos_flat_dim
        
        # Static FC layers — fc_1, fc_2, … (same as TF's first fc loop)
        # These are fixed layers added by the controller (not dynamic new_fc_*).
        static_fc_keys = sorted(
            [k for k in params.keys() if re.match(r'^fc_\d+$', k) and params[k] > 0],
            key=lambda s: int(s.split('_')[-1])
        )
        for i, key in enumerate(static_fc_keys):
            out_dim = int(params[key] * params['num_neurons'])
            self.classifier.add_module(f"sfc_{i}",   nn.Linear(current_dim, out_dim))
            self.classifier.add_module(f"sact_{i}",  copy.deepcopy(act_fn))
            self.classifier.add_module(f"sdrop_{i}", nn.Dropout(params['dr_f']))
            current_dim = out_dim

        # Dynamically added FC layers (new_fc_1, new_fc_2, etc.)
        fc_keys = sorted([k for k in params.keys() if re.match(r'new_fc_\d+', k)],
                         key=lambda x: int(x.split('_')[-1]))
        
        for i, key in enumerate(fc_keys):
            val = params[key]
            if val > 0:
                out_dim = int(val * params['num_neurons'])
                self.classifier.add_module(f"fc_{i}", nn.Linear(current_dim, out_dim))
                self.classifier.add_module(f"act_{i}", act_fn)
                self.classifier.add_module(f"drop_{i}", nn.Dropout(params['dr_f']))
                current_dim = out_dim
                
        # Final Output Layer
        self.classifier.add_module("fc_out", nn.Linear(current_dim, self.n_classes))
        
        self.modules_list = nn.ModuleList()
        for module in self.features:
            self.modules_list.append(module)
        for module in self.classifier:
            self.modules_list.append(module)
            
        self.create_specs()

    def _get_activation(self, name):
        name = name.lower()
        if name == 'elu': return nn.ELU(inplace=True)
        if name == 'selu': return nn.SELU(inplace=True)
        return nn.ReLU(inplace=True)

    def summary(self):
        """
        Print a custom model summary in TensorFlow-style format.
        Displays model architecture, layer information, and parameter count.
        Expands ConvBlock to show internal layers.
        """
        print("=" * 80)
        print("Model Summary - TorchModel")
        print("=" * 80)
        
        # Count total parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        print(f"\nInput shape: {self.input_shape}")
        print(f"Number of classes: {self.n_classes}")
        print(f"Residual connections: {self.use_residual}")
        print(f"Batch normalization: {self.batch}\n")
        
        # Create dummy input and trace through features to get actual shapes
        dummy_input = torch.zeros(1, *self.input_shape)
        
        was_training = self.training
        self.eval()
        
        # Print features layers
        print("Features (Convolutional layers):")
        print("-" * 80)
        print(f"{'Layer (type)':<30} {'Output Shape':<25} {'Param #':<15}")
        print("-" * 80)
        
        with torch.no_grad():
            x = dummy_input
            for name, module in self.features.named_children():
                x = module(x)
                params = sum(p.numel() for p in module.parameters())
                module_name = f"{name} ({module.__class__.__name__})"
                output_shape = f"{tuple(x.shape)}"
                
                # Print the module
                print(f"{module_name:<30} {output_shape:<25} {params:>14,}")
                
                # If it's a ConvBlock, expand and show internal layers
                if isinstance(module, ConvBlock):
                    # Trace through ConvBlock's internal layers
                    x_block = x  # This is the output, but we need to trace from input
                    # We need to re-trace from the input to this block
                    if name == "block_c1":
                        x_in = dummy_input
                    else:
                        # Re-run features up to this point to get input
                        x_in = dummy_input
                        for n2, m2 in self.features.named_children():
                            if n2 == name:
                                break
                            x_in = m2(x_in)
                    
                    with torch.no_grad():
                        x_internal = x_in
                        # Show internal Conv layers
                        for i, layer in enumerate(module.layers):
                            x_internal = layer(x_internal)
                            layer_params = sum(p.numel() for p in layer.parameters())
                            print(f"  ├─ conv_block_{i} (Sequential)     {str(tuple(x_internal.shape)):<25} {layer_params:>14,}")
                        
                        # Show last conv
                        x_internal = module.last_conv(x_internal)
                        last_conv_params = sum(p.numel() for p in module.last_conv.parameters())
                        print(f"  ├─ last_conv (Conv2d)            {str(tuple(x_internal.shape)):<25} {last_conv_params:>14,}")
                        
                        # Show last bn
                        x_internal = module.last_bn(x_internal)
                        last_bn_params = sum(p.numel() for p in module.last_bn.parameters())
                        print(f"  ├─ last_bn (BatchNorm2d)         {str(tuple(x_internal.shape)):<25} {last_bn_params:>14,}")
                        
                        if module.use_residual:
                            # Show shortcut
                            shortcut_params = sum(p.numel() for p in module.shortcut.parameters())
                            print(f"  └─ shortcut (Identity/Conv)      {str(tuple(x_internal.shape)):<25} {shortcut_params:>14,}")
        
        print("-" * 80)
        
        # Compute classifier input shape and trace through it
        print("\nClassifier (Fully Connected layers):")
        print("-" * 80)
        print(f"{'Layer (type)':<30} {'Output Shape':<25} {'Param #':<15}")
        print("-" * 80)
        
        with torch.no_grad():
            x = dummy_input
            x = self.features(x)
            if self.batch:
                x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            # For ROI models, concatenate a zero pos tensor to match classifier input size
            if self.is_roi and self.pos_input_shape is not None:
                dummy_pos = torch.zeros(1, self.pos_flat_dim)
                x = torch.cat([x, dummy_pos], dim=1)
            
            for name, module in self.classifier.named_children():
                x = module(x)
                params = sum(p.numel() for p in module.parameters())
                module_name = f"{name} ({module.__class__.__name__})"
                output_shape = f"{tuple(x.shape)}"
                
                print(f"{module_name:<30} {output_shape:<25} {params:>14,}")
        
        print("-" * 80)
        
        # Summary statistics
        print("\nTotal params: {:,}".format(total_params))
        print("Trainable params: {:,}".format(trainable_params))
        print("Non-trainable params: {:,}".format(non_trainable_params))
        print("=" * 80)
        
        if was_training:
            self.train()


    def forward(self, x, pos=None):
        x = self.features(x)
        if self.batch:
            # Global Average Pooling: (N, C, H, W) -> (N, C, 1, 1)
            x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)                     # (N, flat_dim)
        if self.is_roi and pos is not None:
            # Replicate TF: Flatten(pos_map) then Concatenate with image features
            pos_flat = pos.view(pos.shape[0], -1)    # (N, pos_flat_dim)
            x = torch.cat([x, pos_flat], dim=1)      # (N, flat_dim + pos_flat_dim)
        x = self.classifier(x)
        return x
    
    def create_specs(self):
        """
        Create LayerSpec objects for each module in the network by hooking into
        the forward pass and capturing input/output information for each layer.
        """
        self.eval()
        # input_shape is (C, H, W), so dummy_input should be (batch, C, H, W)
        dummy_input = torch.zeros([1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])

        def make_hook(module_name):
            """Factory function to create a hook that captures layer specifications"""
            def hook_fn(module, input, output):
                self.layers[module_name] = self.to_spec(module_name, module, input, output)

            return hook_fn

        # Register forward hooks to capture layer information
        # Only register hooks on supported layer types, skip custom modules like ConvBlock
        hooks = []
        for name, module in self.modules_list.named_children():
            if not isinstance(module, (TorchModel, ConvBlock)):
                if module.__class__ in self.to_type_map:
                    hooks.append(module.register_forward_hook(make_hook(name)))

        # For ROI models the forward pass requires a pos tensor; provide a zero
        # dummy so the classifier's Linear layer receives the correct input size.
        dummy_pos = None
        if self.is_roi and self.pos_input_shape is not None:
            dummy_pos = torch.zeros([1, *self.pos_input_shape])

        # Perform forward pass to trigger hooks
        self.layers = {}
        with torch.no_grad():
            self(dummy_input, dummy_pos)

        # Clean up hooks
        for hook in hooks:
            hook.remove()


    def from_type(self, layer_type: LayerTypes):
        """
        Get the PyTorch module class for a given layer type.
        
        Args:
            layer_type: Custom layer type enum
            
        Returns:
            Corresponding PyTorch module class
        """
        return self.from_type_map[layer_type]

    def to_type(self, cls: type[nn.Module]):
        """
        Convert a PyTorch module class to its corresponding custom layer type.
        
        Args:
            cls: PyTorch module class
            
        Returns:
            Corresponding custom layer type
        """
        return self.to_type_map[cls]

    def to_spec(self, module_name, module, input, output):
        """
        Convert a PyTorch module into a LayerSpec object based on its type and
        the input/output tensor information.
        
        Args:
            module_name: Name identifier for the module
            module: The PyTorch module to convert
            input: Tuple containing input tensor(s) to the module
            output: Output tensor from the module
            
        Returns:
            LayerSpec object representing the module
        """
        layer_type = self.to_type(module.__class__)
        if layer_type == LayerTypes.Conv2D:
            return LayerSpec(
                name=module_name, 
                type=layer_type,
                module=module,
                params={
                    Params.IN_CHANNELS: input[0].shape[1],
                    Params.IN_HEIGHT: input[0].shape[2],
                    Params.IN_WIDTH: input[0].shape[3],
                    Params.OUT_CHANNELS: output.shape[1],
                    Params.OUT_HEIGHT: output.shape[2],
                    Params.OUT_WIDTH: output.shape[3],
                    Params.KERNEL_SIZE: module.kernel_size,
                    Params.STRIDE: module.stride,
                    Params.PADDING: 'valid' if module.padding[0] == 0 else 'same',
                    Params.BIAS: module.bias is not None
                }
            )
        elif layer_type == LayerTypes.Dense:
            return LayerSpec(
                name=module_name, 
                type=layer_type,
                module=module,
                params={
                    Params.IN_FEATURES: input[0].shape[1],
                    Params.OUT_FEATURES: output.shape[1],
                }
            )
        elif layer_type == LayerTypes.MaxPooling2D:
            return LayerSpec(
                name=module_name, 
                type=layer_type,
                module=module,
                params={
                    Params.KERNEL_SIZE: module.kernel_size,
                    Params.STRIDE: module.stride
                }
            )
        elif layer_type == LayerTypes.Dropout:
            return LayerSpec(
                name=module_name, 
                type=layer_type,
                module=module,
                params={
                    Params.DROPOUT_RATE: module.p
                }
            )
        # Activation function layers
        elif layer_type in [LayerTypes.ELU, LayerTypes.ReLU, LayerTypes.Sigmoid, LayerTypes.SiLU, LayerTypes.SeLU]:
            return LayerSpec(
                name=module_name, 
                type=LayerTypes.Activation,
                module=module,
                is_activation=True,
                params={
                    "activation_type": layer_type
                }
            )
        # Batch normalization layers
        elif layer_type in [LayerTypes.BatchNormalization1D, LayerTypes.BatchNormalization2D]:
            return LayerSpec(
                name=module_name,
                type=layer_type,
                module=module,
                params={
                    Params.NUM_FEATURES: module.num_features
                }
            )
        elif layer_type == LayerTypes.Flatten:
            return LayerSpec(
                name=module_name,
                type=layer_type,
                module=module,
                params={
                    "start_dim": module.start_dim,
                    "end_dim": module.end_dim
                }
            )
        else:
            raise Exception("Missing LayerSpec for layer of type " + module.__class__.__name__)

    def from_spec(self, layer_spec: LayerSpec):
        """
        Create a PyTorch module from a LayerSpec object.
        
        Args:
            layer_spec: Layer specification containing type and parameters
            
        Returns:
            Initialized PyTorch module
        """
        if layer_spec.type == LayerTypes.Conv2D:
            return nn.Conv2d(
                in_channels=layer_spec.get(Params.IN_CHANNELS),
                out_channels=layer_spec.get(Params.OUT_CHANNELS),
                kernel_size=layer_spec.get(Params.KERNEL_SIZE),
                padding=layer_spec.get(Params.PADDING),
            )
        elif layer_spec.type == LayerTypes.Dense:
            return nn.Linear(
                in_features=layer_spec.get('in_features'),
                out_features=layer_spec.get('out_features')
            )
        elif layer_spec.type == LayerTypes.MaxPooling2D:
            return nn.MaxPool2d(
                kernel_size=layer_spec.get(Params.KERNEL_SIZE),
                stride=layer_spec.get(Params.STRIDE)
            )
        elif layer_spec.type == LayerTypes.Dropout:
            return nn.Dropout(
                p=layer_spec.get(Params.DROPOUT_RATE)
            )
        elif layer_spec.type in [LayerTypes.BatchNormalization1D, LayerTypes.BatchNormalization2D]:
            return self.from_type(layer_spec.type)(
                num_features=layer_spec.get(Params.NUM_FEATURES)
            )
        elif layer_spec.type == LayerTypes.Flatten:
            return nn.Flatten(
                start_dim=layer_spec.get("start_dim"),
                end_dim=layer_spec.get("end_dim")
            )
        elif layer_spec.type == LayerTypes.Activation:
            return self.from_type(layer_spec.get("activation_type"))()
        else:
            return self.from_type(layer_spec.type)()
