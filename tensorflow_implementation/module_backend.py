import numpy as np
from tensorflow.keras import layers, models
from components.backend_interface import BackendInterface

class ModuleBackend(BackendInterface):

    def get_layers(self, model):
        return model.layers

    def get_input_shape(self, layer):
        return layer.input.shape

    def get_output_shape(self, layer):
        return layer.output.shape

    def get_layer_info(self, layer):
        """Translate TensorFlow layers to common standard types"""
        tf_layer_type = layer.__class__.__name__

        if tf_layer_type == 'Conv2D':
            return 'conv2d', {
                'in_channels': layer.input.shape[-1],
                'out_channels': layer.output.shape[-1],
                'kernel_size': layer.kernel_size[0],
                'stride': layer.strides[0],
                'padding': 'valid' if layer.padding == 'valid' else 'same',
                'bias': layer.use_bias
            }
        elif tf_layer_type == 'Dense':
            return 'dense', {
                'in_features': layer.input.shape[-1],
                'out_features': layer.output.shape[-1],
                'bias': layer.use_bias
            }
        else:
            return 'other', {}  # unknown / ignored layers
        
    def get_flops(self, model, input_shapes=None):
        from tensorflow_implementation.flops.flops_calculator import analyze_model
        flops = analyze_model(model, input_shapes)[0].total_float_ops
        try:
            trainableParams = np.sum([np.prod(v.shape) for v in model.model.trainable_weights])
            nonTrainableParams = np.sum([np.prod(v.shape) for v in model.model.non_trainable_weights])
        except AttributeError:
            trainableParams = np.sum([np.prod(v.shape) for v in model.trainable_weights])
            nonTrainableParams = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
        nparams = trainableParams + nonTrainableParams
        return flops, nparams
    
    def get_params(self, model):
        trainableParams = np.sum([np.prod(v.shape) for v in model.model.trainable_weights])
        nonTrainableParams = np.sum([np.prod(v.shape) for v in model.model.non_trainable_weights])
        nparams = trainableParams + nonTrainableParams
        return nparams