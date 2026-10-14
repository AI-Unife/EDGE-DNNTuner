from time import time
from typing import Optional

import numpy as np


from components.search_space import search_space
from exp_config import load_cfg

from abc import ABC, abstractmethod

from components.dataset import TunerDataset
from components.backend_interface import BackendInterface

class NeuralNetwork (ABC):
    """
    class used for the management of the neural network architecture,
    offering methods for training the dnn and adding and removing convolutional layers
    """
    def __init__(self, backend: BackendInterface, dataset: TunerDataset, da: bool, reg: bool, residual: bool):
        """
        initialized the attributes of the class.
        first part is used for storing the examples of the dataset,
        the second part to keep track of the number of the various parts of the dnn,
        for example the number of convolutional or dense layers.
        """
        
        self.dataset = dataset
        self.backend = backend
        
        self.dataset.data_as_float32()

        self.exp_cfg = load_cfg()
        self.n_classes = self.dataset.n_classes
        self.epochs = self.exp_cfg.epochs
        # Legacy flags (kept for API compatibility)
        self.last_dense = 0
        self.counter_fc = 0
        self.counter_conv = 0
        self.rgl = False
        self.dense = False
        self.conv = False

        # Populated during training
        self.flops: Optional[float] = None
        self.nparams: Optional[float] = None
        self.tot_latency_cost: Optional[float] = None

        self.da = da
        self.reg = reg
        self.residual = residual
        self.model = None
    
        

    def _json_default(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    
    @abstractmethod
    def build_network(self, params, layer_x_block):
        raise NotImplementedError


    @abstractmethod
    def training(self, params):
        """
        Function for compiling and running training
        :param params: parameters to indicate a possible operation on the network structure and hyperparameter search space
        :return: training history, trained model and and performance evaluation score 
        """
        raise NotImplementedError
    
    @abstractmethod
    def save_model(self, model, model_name_id):
        """
        Function for saving the model in a json file
        :param model: model to be saved
        :param model_name_id: unique identifier for the model name
        :return: -
        """
        raise NotImplementedError
    
    @abstractmethod
    def eval_model(self):
        """
        Function for evaluating the model performance
        :return: evaluation score
        """
        raise NotImplementedError

