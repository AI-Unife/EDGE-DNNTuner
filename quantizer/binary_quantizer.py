
import tensorflow as tf

#### Define here your quantizer class, inheriting from quantizer_interface
#### rename this file with the name of your quantizer 


from quantizer.quantizer_interface import quantizer_interface
import tensorflow as tf
from exp_config import load_cfg


class quantizer_module(quantizer_interface):
    
    def __init__(self, opt):
        self.opt = opt
        self.quantized_model = None
        ### ADD this part to load configuration file
        self.cfg = load_cfg()

    def binarize_weights_2(self, model):
        """
        Binarize the weights of a Keras model.
        Replaces values <= 0 with -1 and values > 0 with 1.
        """
        weights = model.get_weights()
        modified_weights = []
        for weight_matrix in weights:
            # tf.where also works on numpy arrays as they get converted
            modified_matrix = tf.where(weight_matrix <= 0, -1.0, 1.0)
            modified_weights.append(modified_matrix.numpy())  # assicurati di restituire np
        return modified_weights

    def quantizer_function(self, model):
        # binarize the weights of the original model
        binarized_weights = self.binarize_weights_2(model)

        # clone the model
        self.quantized_model = tf.keras.models.clone_model(model)

        # compile with the same loss as the original model (e.g., binary)
        self.quantized_model.compile(
            optimizer=self.opt,
            loss="binary_crossentropy",
            metrics=['accuracy']
        )

        # set the binarized weights
        self.quantized_model.set_weights(binarized_weights)

        return self.quantized_model
    
    def evaluate_quantized_model(self, x_test, y_test):
        if self.quantized_model is None:
            raise ValueError("Quantized model is not available. Please run quantize_function first.")
        
        score = self.quantized_model.evaluate(x_test, y_test, verbose=2)
        return score

    def save_quantized_model(self, path="quantized_model.keras"):
        self.quantized_model.save(path)

    def printing_values(self):
        pass

    def plotting_function(self):
        pass

    def log_function(self):
        pass



if __name__ == "__main__":
    ### An example of how to use this quantizer module
    ### You can run this cell to test your quantizer module
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten
    
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the pixel values to be between 0 and 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Filter the dataset to include only digits 0 and 1.
    # In my test I used this cell to make faster the train for test purpouse.
    def filter_zeros_ones(x, y):
        keep = (y == 0) | (y == 1)
        x = x[keep]
        y = y[keep]
        return x, y

    x_train_filtered, y_train_filtered = filter_zeros_ones(x_train, y_train)
    x_test_filtered, y_test_filtered = filter_zeros_ones(x_test, y_test)

    selected = 3

    print(f"Original training data shape: {x_train.shape}, {y_train.shape}")
    print(f"Filtered training data shape: {x_train_filtered.shape}, {y_train_filtered.shape}")
    print(f"Original test data shape: {x_test.shape}, {y_test.shape}")
    print(f"Filtered test data shape: {x_test_filtered.shape}, {y_test_filtered.shape}")
    
    # Build the Dense network model
    model_FC = Sequential([
        Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images into a 1D array
        Dense(8, activation='relu'), # Add a dense layer with 128 units and ReLU activation
        Dense(1, activation='sigmoid') # Add an output layer with 10 units (for 10 classes) and softmax activation
    ])

    # Compile the model
    model_FC.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    model_FC.summary()

    # Define Early Stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    model_FC.fit(x_train_filtered, y_train_filtered, epochs=50, validation_data=(x_test_filtered, y_test_filtered), callbacks=[early_stopping])

    # Evaluate the model on the test set
    loss_model_FC, accuracy_model_FC = model_FC.evaluate(x_test_filtered, y_test_filtered)
    print(f"Test loss: {loss_model_FC}")
    print(f"Test accuracy: {accuracy_model_FC}")
    
    quantizer = quantizer_module(opt='adam')
    quantized_model = quantizer.quantizer_function(model_FC)
    score = quantizer.evaluate_quantized_model(x_test_filtered, y_test_filtered)
    print(f"Test loss quantizer: {score[0]}")
    print(f"Test accuracy quantizer: {score[1]}")