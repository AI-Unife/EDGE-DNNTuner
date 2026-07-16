from datetime import datetime
import os

import torch
from typing import Any, List, Tuple, Dict
import re

import copy
import numpy as np

from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from torch.nn import functional as F

from components.model_interface import LayerTypes, TunerModel
from components.neural_network import NeuralNetwork as BaseNeuralNetwork
from components.dataset import TunerDataset
from components.backend_interface import BackendInterface
from pytorch_implementation.model import TorchModel


class TorchTunerDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor,
                 pos: torch.Tensor = None, transform=None):
        self.images = images
        self.labels = labels
        self.pos = pos          # None for non-ROI datasets
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        image = self.images[index]
        if self.transform:
            image = self.transform(image)
        if self.pos is not None:
            return image, self.labels[index], self.pos[index]
        return image, self.labels[index]
    

class NeuralNetwork(BaseNeuralNetwork):

    def __init__(self, backend:BackendInterface, dataset: TunerDataset, da: bool, reg: bool, residual: bool):
        super().__init__(backend, dataset, da, reg, residual)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Framework-specific preprocessing
        if self.dataset.X_train.ndim == 3:
            self.dataset.X_train = self.dataset.X_train[..., None]
            self.dataset.X_test = self.dataset.X_test[..., None]

        self.train_data = self.dataset.X_train
        self.test_data = self.dataset.X_test

        self.train_images = self.to_tensor(self.train_data)
        self.test_images = self.to_tensor(self.test_data)
        # Y_train may be:
        #   [N]          integer labels  (standard datasets)
        #   [N, C]       one-hot         (TF-converted datasets)
        #   [N, T, C]    temporal one-hot (gesture ToOneHotTimeCoding)
        # PyTorch CrossEntropyLoss always needs 1-D integer class indices [N].
        self.train_labels = torch.from_numpy(
            self._extract_int_labels(self.dataset.Y_train)
        ).long()
        self.test_labels = torch.from_numpy(
            self._extract_int_labels(self.dataset.Y_test)
        ).long()

        # Convert position maps to tensors for ROI datasets.
        # pos arrays have shape (N, T, H, W, C) or (N, C, H, W) depending on mode;
        # keep the original numpy layout and convert to float32 tensor without
        # reordering axes — the model flattens them entirely.
        if self.is_roi and hasattr(self.dataset, 'pos_train') and self.dataset.pos_train is not None:
            self.train_pos = torch.from_numpy(
                self.dataset.pos_train.astype('float32')
            )
            self.test_pos = torch.from_numpy(
                self.dataset.pos_test.astype('float32')
            )
        else:
            self.train_pos = None
            self.test_pos = None

        self.activation_map = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "selu": nn.SELU,
            "swish": nn.SiLU
        }

        # Map optimizers
        self.optimizer_map = {
            "Adam": optim.Adam,
            "Adamax": optim.Adamax,
            "Adagrad": optim.Adagrad,
            "Adadelta": optim.Adadelta,
            "RMSprop": optim.RMSprop,
            "SGD": optim.SGD
        }
    
    @staticmethod
    def _extract_int_labels(y: np.ndarray) -> np.ndarray:
        """
        Convert any label format to a 1-D integer class-index array [N].

        Handles:
          y.ndim == 1  → already integer [N], return as-is
          y.ndim == 2  → one-hot [N, C], take argmax over last axis
          y.ndim == 3  → temporal one-hot [N, T, C] (ToOneHotTimeCoding),
                         all frames carry the same label so take frame 0
        """
        if y.ndim == 1:
            return y.astype(np.int64)
        elif y.ndim == 2:
            return np.argmax(y, axis=-1).astype(np.int64)
        elif y.ndim == 3:
            return np.argmax(y[:, 0, :], axis=-1).astype(np.int64)
        else:
            raise ValueError(f"Unsupported label shape: {y.shape}")

    @staticmethod
    def to_tensor(array):
        if array.ndim == 3:
            # (N, H, W) → (N, H, W, 1)
            array = array[..., None]
        t = torch.from_numpy(array)
        if t.ndim == 4:
            # (N, H, W, C) → (N, C, H, W)
            return t.permute(0, 3, 1, 2).contiguous().float()
        elif t.ndim == 5:
            # Temporal gesture: (N, T, H, W, C) → (N, T, C, H, W)
            return t.permute(0, 1, 4, 2, 3).contiguous().float()
        else:
            return t.contiguous().float()
    
    def _model_forward(self, inputs: torch.Tensor, pos: torch.Tensor = None) -> torch.Tensor:
        """
        Run a forward pass through the model, handling both standard 4D input
        (N, C, H, W) and temporal 5D input (N, T, C, H, W).

        For temporal data each frame is processed independently; the per-frame
        logits are averaged across the time dimension before returning, so the
        output is always (N, n_classes) regardless of the number of frames.

        pos is the flattened position map (N, *pos_shape) for ROI datasets;
        it is forwarded to TorchModel.forward which concatenates it to features.
        """
        if inputs.ndim == 5:
            N, T, C, H, W = inputs.shape
            # Flatten the time dimension into the batch dimension
            flat = inputs.view(N * T, C, H, W)
            # Expand pos along the time axis so each frame gets its own pos slice
            pos_flat = None
            if pos is not None:
                # pos shape: (N, T, ...) or (N, ...) — expand to (N*T, ...)
                if pos.ndim >= 2 and pos.shape[1] == T:
                    pos_flat = pos.view(N * T, *pos.shape[2:])
                else:
                    pos_flat = pos.unsqueeze(1).expand(N, T, *pos.shape[1:]).reshape(N * T, *pos.shape[1:])
            out_flat = self.model(flat, pos_flat)    # (N*T, n_classes)
            return out_flat.view(N, T, -1).mean(dim=1)
        return self.model(inputs, pos)

    def build_network(self, params, layer_x_block=2):
        """
        Build the PyTorch model according to the given hyperparameters.
        """
        if self.dataset.X_train.ndim == 5:
            # Temporal dataset: (N, T, H, W, C) — the model operates on single
            # frames, so derive the per-frame shape (H, W, C) from axis 2 onward.
            frame_shape = self.dataset.X_train.shape[2:]  # (H, W, C)
            self.input_shape = (frame_shape[2], frame_shape[0], frame_shape[1])  # (C, H, W)
        else:
            input_shape = self.dataset.X_train.shape[1:]  # (H, W, C)
            self.input_shape = (input_shape[2], input_shape[0], input_shape[1])  # (C, H, W)

        # Match TF logic: BatchNorm in conv blocks only for tiny-imagenet and cim datasets
        dataset_name = self.exp_cfg.dataset.lower()
        use_bn = "tiny" in dataset_name or "cim" in dataset_name

        # Derive pos_input_shape from the dataset for ROI models.
        # Must match TF's logic: for temporal data (5D images) TF uses shape[2:]
        # to get the per-frame pos shape; for single-frame data (4D) it uses shape[1:].
        pos_input_shape = None
        if self.is_roi and self.train_pos is not None:
            if self.train_images.ndim == 5:
                # Temporal dataset: train_images (N, T, C, H, W), train_pos (N, T, ...)
                # → per-frame pos shape = shape[2:]  (drop batch and time dims)
                pos_input_shape = tuple(self.train_pos.shape[2:])
            else:
                # Single-frame dataset: train_images (N, C, H, W), train_pos (N, ...)
                # → pos shape = shape[1:]  (drop batch dim only)
                pos_input_shape = tuple(self.train_pos.shape[1:])

        self.model = TorchModel(
            params=params,
            input_shape=self.input_shape,
            n_classes=self.dataset.n_classes,
            layer_x_block=layer_x_block,
            batch=use_bn,
            is_roi=self.is_roi,
            pos_input_shape=pos_input_shape,
        ).to(self.device)

        print("Model Summary:")
        self.model.summary()

        # Collect parameters for L2 regularization if needed
        self.l2_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if "flops_module" in self.exp_cfg.mod_list:
            # Compute FLOPs (approximate; counts MACs as 2 FLOPs)

            self.flops, self.nparams = self.backend.get_flops(self.model, self.input_shape)

        if "hardware_module" in self.exp_cfg.mod_list:
            # Compute total latency cost
            from modules.loss.hardware_module import hardware_module
            HW_module = hardware_module(weight_cost=0.7)
            HW_module.update_state(self.model)
            self.tot_latency_cost = HW_module.total_cost
 
        

        return self.model

    def training(self, params: Dict[str, Any]) -> Tuple[List[float], Dict[str, List[float]], TunerModel]:
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
        if self.model is None:
            print("Error: Model is not built.")
            exit(1)

            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model_name_id = datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")

        print(f"Training model {model_name_id} on device: {self.device}")

        # Try loading previous weights if available (fine-tune / warm start)
        try:
            prev_weights = f"{self.exp_cfg.name}/Weights/weights.h5"
            if os.path.exists(prev_weights):
                self.model.load_weights(prev_weights)
        except Exception:
            pass  # ignore if incompatible
        
        # 1. Setup Ottimizzatore da params
        lr = params['learning_rate']
        opt_name = params['optimizer'].lower()
        batch_size = int(params['batch_size'])
        
        if opt_name == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        elif opt_name == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif opt_name == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=lr)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=lr) # fallback
            
        self.criterion = nn.CrossEntropyLoss()
        self.model.optimizer = optimizer
        
            
        
        # Layer wise learning rate
        parameters = []
        current_mul = 1
        lr_factor = 1.414213

        for module in self.model.modules_list:
            trainable_parameters = [p for n, p in module.named_parameters() if p.requires_grad]

            if not len(trainable_parameters):
                continue

            if module.type == LayerTypes.Conv2D:
                lr = params["learning_rate"] * current_mul
                current_mul /= lr_factor
            else:
                lr = params["learning_rate"]

            parameters += [{
                'params': trainable_parameters,
                'lr': lr
            }]

        optimizer = self.optimizer_map[params['optimizer']](parameters)

        # Learning rate adjustment
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-4)

        # Data augmentation
        if self.da:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ])
        else:
            transform = None

        # Pass pos tensors to the dataset when ROI is active so the loader
        # returns (inputs, labels, pos) triplets instead of (inputs, labels) pairs.
        train_dataset = TorchTunerDataset(
            self.train_images, self.train_labels,
            pos=self.train_pos, transform=transform
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        test_dataset = TorchTunerDataset(
            self.test_images, self.test_labels,
            pos=self.test_pos, transform=None
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Training loop
        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

        best_val_loss = float('inf')
        best_val_acc = -float('inf')
        patience = 15
        min_delta = 0.005
        counter_loss = 0
        counter_acc = 0
        
        
        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        for epoch in range(self.exp_cfg.epochs):
            # Check if "debug" is in the experiment name
            if 'debug' in self.exp_cfg.name.lower():
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
                
            else:
                self.model.train()
                running_loss, correct = 0.0, 0
                for batch in train_loader:
                    # Unpack: (inputs, labels) for standard datasets,
                    #         (inputs, labels, pos) for ROI datasets
                    if len(batch) == 3:
                        inputs, labels, pos = batch
                        pos = pos.to(self.device)
                    else:
                        inputs, labels = batch
                        pos = None
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    # _model_forward handles 4D/5D inputs and optional pos map
                    outputs = self._model_forward(inputs, pos)
                    loss = self.criterion(outputs, labels)
                    # Regularization
                    if self.rgl:
                        l2 = sum(p.pow(2).sum() for p in self.l2_params)
                        loss = loss + params['reg'] * l2
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    correct += (outputs.argmax(1) == labels).sum().item()

                train_loss = running_loss / len(train_loader)
                train_acc = correct / len(self.train_labels)
                score = self.eval_model(self.model, test_loader)
                val_loss, val_acc = score[0], score[1]

                # Update history
                history['loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['accuracy'].append(train_acc)
                history['val_accuracy'].append(val_acc)

            print(f"Epoch {epoch+1}: loss={train_loss:.4f}, val_loss={val_loss:.4f}, acc={train_acc:.4f}, val_acc={val_acc:.4f}")

            scheduler.step(val_loss)

            # Early stopping on val_loss (mode='min')
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                counter_loss = 0
                self.save_model(params)
                best_model_wts = copy.deepcopy(self.model.state_dict()) # Copia in RAM
                saved = True
                print(f"  -> Model Saved (Best Loss: {best_val_loss:.4f})")
            else:
                counter_loss += 1

            # Early stopping on val_acc (mode='max')
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                counter_acc = 0
                if not saved: # Avoid double save if already saved for loss
                    self.save_model(params)
                    print(f"  -> Model Saved (Best Acc: {best_val_acc:.4f})")
            else:
                counter_acc += 1

            # Stop if either condition reaches patience
            if counter_loss >= patience or counter_acc >= patience:
                print("Early stopping triggered.")
                break
        
        print("Loading best model weights...")
        self.model.load_state_dict(best_model_wts)
        
        self.save_model(params)
        
        return [best_val_loss, best_val_acc], history, self.model
    
    
    def eval_model(self, model: TunerModel, test_loader) -> Tuple[float, float]:
        """
        Evaluate the model on the test set. Returns loss and accuracy.
        """
        # Validation
        self.model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 3:
                    inputs, labels, pos = batch
                    pos = pos.to(self.device)
                else:
                    inputs, labels = batch
                    pos = None
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # _model_forward handles 4D/5D inputs and optional pos map
                outputs = self._model_forward(inputs, pos)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_loss /= len(test_loader)
        val_acc = val_correct / len(self.test_labels)
        score = [val_loss, val_acc]
        return score

    def save_model(self, params=None):
        """
        Saves the model weights and the architecture configuration to a single .pth file.
        This replicates Keras' ability to save 'everything' needed to run the model later.
        """
        if self.model is None:
            return

        try:
            # 1. Define the directory and create it if it doesn't exist
            save_dir = os.path.join(self.exp_cfg.name, "Model")
            os.makedirs(save_dir, exist_ok=True)
            
            # 2. Define the full file path
            file_path = os.path.join(save_dir, "best_model.pth")

            # 3. Create a dictionary containing EVERYTHING needed to reconstruct the model
            checkpoint = {
                # Architecture parameters (CRITICAL for DynamicNet)
                'params': params,  
                
                # Model dimensions
                'input_shape': self.input_shape, 
                'num_classes': self.dataset.n_classes,           
                
                # The actual learned weights
                'model_state_dict': self.model.state_dict(),
                
                # (Optional) Optimizer state if you want to resume training later
                # 'optimizer_state_dict': self.optimizer.state_dict()
            }
            
            # 4. Save the checkpoint dictionary to disk
            torch.save(checkpoint, file_path)
            print(f"-> Best model successfully saved to: {file_path}")

        except Exception as e:
            print(f"[ERROR] Failed to save best model: {e}")
            
    ### TODO: da sistemare
    def load_network(self, file_path):
        """
        Loads a DynamicNet from a .pth checkpoint.
        
        Args:
            file_path (str): Path to the .pth file.
            device (str): 'cpu' or 'cuda'.
            
        Returns:
            model (nn.Module): The reconstructed and loaded model, set to eval mode.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No model found at {file_path}")

        print(f"Loading model from {file_path}...")
        
        # 1. Load the checkpoint dictionary
        # map_location ensures we can load a GPU model on CPU if needed
        checkpoint = torch.load(file_path, map_location=self.device)
        
        # 2. Extract configuration
        params = checkpoint['params']
        input_shape = checkpoint['input_shape']
        num_classes = checkpoint['num_classes']
        
        # 3. Instantiate the "empty" DynamicNet architecture
        # This rebuilds the exact structure (layers, neurons) used during training
        model = DynamicNet(params, input_shape, num_classes)
        
        # 4. Load the weights into the architecture
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 5. Move to device and set to evaluation mode (freezes BatchNorm/Dropout)
        model.to(self.device)
        model.eval()
        
        return model