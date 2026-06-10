from tensorflow import keras
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow_implementation.flops.node_manager as nm
import sys


class flop_calculator:
    """
    class used to calculate the number of flops of a neural network
    """
    def flop_calc(self, concrete_func):
        """
        method used to calculate flops
        :param concrete_func: concrete function
        :return: final number of flops
        """
        try:
            # replaces variables in a graph with constants of the same values
            _, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
        except ValueError as e:
            # Fallback for stateful RNG/resource-handle conversion issues on some TF builds.
            if "incompatible with expected resource" not in str(e):
                raise
            graph_def = concrete_func.graph.as_graph_def()

        # operate on the graph
        with tf.Graph().as_default() as graph:
            # imports the graph from graph_def
            tf.graph_util.import_graph_def(graph_def, name='')

            # used to extract secondary informations runtime, such as execution time and memory consumption
            run_meta = tf.compat.v1.RunMetadata()

            # build profiler to monitor a given network parameter, in this case 'float operations'
            opts = (tf.compat.v1.profiler.ProfileOptionBuilder(
            ).with_empty_output().float_operation())
            opts['output'] = 'none'

            # get number of flops through the profiler
            flops = tf.compat.v1.profiler.profile(
                graph=graph, run_meta=run_meta, cmd="op", options=opts)

            return flops

    def get_flops(self, model):
        """
        method used to obtain the number of flops
        :param model: model whose flops need to be calculated
        :return: number of total flops
        """

        # concrete function to apply to each layer to obtain inputs value
        concrete = tf.function(lambda *inputs: model.model(*inputs))
        specs = [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs]
        concrete_func = concrete.get_concrete_function(*specs)

        flops = self.flop_calc(concrete_func)
        res_dict = nm.to_dict(flops)
        print(" ---- REPORT CUSTOM ------\n")
        print('TOTAL_FLOPS: {}\n'.format(flops.total_float_ops))
        print(res_dict)
        return flops.total_float_ops

# initial_model = keras.Sequential([
#         keras.Input(shape=(1, 320, 280, 1)),
#     layers.Conv2D(1, (2, 1), strides=(1, 1), padding="valid",
#                   use_bias=False, kernel_initializer=tf.keras.initializers.Ones())
# ])


def analyze_model(initial_model, input_shapes=None):
    """
    Analyze the model's FLOPs using the provided input shapes.
    
    :param initial_model: model whose flops need to be calculated
    :param input_shapes: list of input shapes (optional, defaults to model.inputs)
    :return: number of total flops
    """
    model = initial_model

    # Use provided input_shapes if available, otherwise infer from model
    if input_shapes is None:
        specs = [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in initial_model.model.inputs]
    else:
        specs = [tf.TensorSpec([1, *shape]) for shape in input_shapes]
    
    # Create concrete function with correct number of inputs
    if len(specs) == 1:
        concrete = tf.function(lambda x: model.model(x), autograph=False)
    else:
        # For multiple inputs (e.g., dual ROI inputs), pass as a list
        def call_model(*inputs):
            return model.model(list(inputs))
        concrete = tf.function(call_model, autograph=False)
    
    concrete_func = concrete.get_concrete_function(*specs)

    # total_flops, graph_nodes = get_flops(concrete_func)

    # calculate number of flops using the previous concrete function
    flops = flop_calculator().flop_calc(concrete_func)

    # convert frozen graph to dict, containing flops number for each operations in the neural network
    res_dict = nm.to_dict(flops)

    return flops, res_dict


if __name__ == "__main__":

    im = VGG16()

    f, r_dict = analyze_model(im)

    print(" ---- REPORT CUSTOM ------\n")
    print('TOTAL_FLOPS: {}\n'.format(f.total_float_ops))
    print(r_dict)
