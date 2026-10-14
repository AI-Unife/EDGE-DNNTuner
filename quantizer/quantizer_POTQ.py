import tensorflow as tf

#### Define here your quantizer class, inheriting from quantizer_interface
#### rename this file with the name of your quantizer 


from quantizer.quantizer_interface import quantizer_interface
import tensorflow as tf
import numpy as np
from exp_config import load_cfg

class quantizer_module(quantizer_interface):

    def __init__(self, opt='adam', n_bits=3):
        """
        Args:
            opt: Optimizer to use for compiling the model.
            n_bits: Number of bits for POTQ quantization.
        """
        self.opt = opt
        self.n_bits = n_bits
        self.quantized_model = None
        ### ADD this part to load configuration file
        self.cfg = load_cfg()
        self.score = 0

    def _potlq_quantizer_numpy(self, weights_flat, q, n_bits):
        """
        Applies the Power-of-Two Level Quantizer to a flattened weight array (NumPy version).
        """
        if q <= 0:
            return np.zeros_like(weights_flat)

        # Define the positive quantization levels
        num_positive_levels = 2 ** (n_bits - 1)
        # equivalent to q * (2 ** arange)
        y_pos = q * (2.0 ** np.arange(num_positive_levels))

        # Define the thresholds between positive levels
        t_pos_mid = (y_pos[:-1] + y_pos[1:]) / 2.0

        # Concatenate 0, midpoints, and infinity
        t_pos = np.concatenate([
            np.array([0.0]),
            t_pos_mid,
            np.array([np.inf])
        ])

        # Define the negative quantization levels and thresholds
        # Flip order and negate
        y_neg = -np.flip(y_pos)
        t_neg = -np.flip(t_pos)

        quantized_weights = np.zeros_like(weights_flat)

        # Quantize positive values
        for i in range(num_positive_levels):
            # Boolean indexing in numpy
            indices = (weights_flat >= t_pos[i]) & (weights_flat < t_pos[i + 1])
            quantized_weights[indices] = y_pos[i]

        # Quantize negative values
        for i in range(num_positive_levels):
            indices = (weights_flat > t_neg[i]) & (weights_flat <= t_neg[i + 1])
            quantized_weights[indices] = y_neg[i]

        return quantized_weights

    def _find_optimal_q_numpy(self, weights, n_bits, num_candidates=100):
        """
        Finds the optimal 'q' for a weight tensor by minimizing Mean Squared Error (NumPy version).
        """
        weights_flat = weights.flatten()
        max_val = np.max(np.abs(weights_flat))

        # Handle case where all weights are zero
        if max_val == 0:
            return 0.0

        # Search for the best q in a reasonable range
        # np.linspace instead of torch.linspace
        limit = max_val / (2 ** (n_bits - 1))
        q_candidates = np.linspace(1e-5, limit, num_candidates)

        min_mse = float('inf')
        best_q = 0.0

        for q_candidate in q_candidates:
            quantized_weights = self._potlq_quantizer_numpy(weights_flat, q_candidate, n_bits)
            # MSE calculation
            mse = np.mean((weights_flat - quantized_weights) ** 2)
            if mse < min_mse:
                min_mse = mse
                best_q = q_candidate

        return best_q

    def quantizer_function(self, model):
        """
        Main function to apply POTQ to the Keras model.
        """
        print(f"\n--- Starting Post-Training PoTLQ ({self.n_bits}-bit) Quantization ---")

        # Get all weights from the model (list of numpy arrays)
        weights = model.get_weights()
        new_weights = []

        for i, w in enumerate(weights):
            # Keras weights:
            # Dense Kernel is usually 2D (input, output)
            # Bias is usually 1D
            # The logic provided says: "Quantize only weight tensors (typically have more than 1 dimension)"

            if w.ndim > 1:
                print(f"  - Quantizing layer index {i} with shape {w.shape}")

                # Find optimal q
                optimal_q = self._find_optimal_q_numpy(w, self.n_bits)
                print(f"    Optimal q found: {optimal_q:.5f}")

                # Quantize
                w_flat = w.flatten()
                w_quantized_flat = self._potlq_quantizer_numpy(w_flat, optimal_q, self.n_bits)

                # Reshape back to original shape
                new_weights.append(w_quantized_flat.reshape(w.shape))
            else:
                # If it's a bias (1D) or scalar, we keep it as is (Float)
                # or you can choose to quantize it too. Usually biases are kept high precision.
                print(f"  - Skipping quantization for layer index {i} (Shape {w.shape} - likely bias)")
                new_weights.append(w)

        # Clone the model structure
        self.quantized_model = tf.keras.models.clone_model(model)

        # Compile the model
        self.quantized_model.compile(
            optimizer=self.opt,
            loss="binary_crossentropy",  # Assuming binary task based on example
            metrics=['accuracy']
        )

        # Set the new quantized weights
        self.quantized_model.set_weights(new_weights)

        print("--- PoTLQ Quantization Complete ---")
        return self.quantized_model

    def evaluate_quantized_model(self, x_test, y_test):
        if self.quantized_model is None:
            raise ValueError("Quantized model is not available. Please run quantize_function first.")
                
        ### ADD this part to manage gesture dataset evaluation
        if (self.cfg.mode in ("fwdPass", "hybrid")) and "gesture" in self.cfg.dataset:
            from tensorflow_implementation.custom_train import eval_model
            self.score = eval_model(self.quantized_model, x_test, y_test)
        else:
            self.score = self.quantized_model.evaluate(x_test, y_test, verbose=2)
        return self.score

    def save_quantized_model(self, path="quantized_model_potq.keras"):
        if self.quantized_model:
            self.quantized_model.save(path)
            print(f"Model saved to {path}")

    def printing_values(self):
        pass

    def plotting_function(self):
        pass

    def log_function(self):
        f = open("{}/algorithm_logs/quantization_report.txt".format(self.cfg.name), "a")
        f.write(str(self.score[0]) + ","+ str(self.score[1])+ "\n")
        f.close()


if __name__ == "__main__":
    ### Test section adapted for POTQ
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten

    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize
    x_train = x_train / 255.0
    x_test = x_test / 255.0


    # Filter dataset (0 and 1 only)
    def filter_zeros_ones(x, y):
        keep = (y == 0) | (y == 1)
        x = x[keep]
        y = y[keep]
        return x, y


    x_train_filtered, y_train_filtered = filter_zeros_ones(x_train, y_train)
    x_test_filtered, y_test_filtered = filter_zeros_ones(x_test, y_test)

    print(f"Original training data shape: {x_train_filtered.shape}")

    # Build the Dense network model
    model_FC = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model_FC.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model briefly
    print("\nTraining original float model...")
    model_FC.fit(x_train_filtered, y_train_filtered, epochs=5, verbose=1)

    # Evaluate original
    loss_model_FC, accuracy_model_FC = model_FC.evaluate(x_test_filtered, y_test_filtered, verbose=0)
    print(f"Original Float Model -> Loss: {loss_model_FC:.4f}, Accuracy: {accuracy_model_FC:.4f}")

    # --- POTQ QUANTIZATION TEST ---
    # Instantiate the quantizer with n_bits=4
    n_bits = 4
    quantizer = quantizer_module(opt='adam', n_bits=n_bits)

    # Run Quantization
    quantized_model = quantizer.quantizer_function(model_FC)

    # Evaluate Quantized
    score = quantizer.evaluate_quantized_model(x_test_filtered, y_test_filtered)
    print(f"\nPOTQ ({n_bits}-bit) Quantized Model -> Loss: {score[0]:.4f}, Accuracy: {score[1]:.4f}")

    # Save
    quantizer.save_quantized_model("quantized_potq_test.keras")