### An example of how to use this quantizer module
### You can run this cell to test your quantizer module
from components.dataset import get_datasets
from components.neural_network import neural_network
from exp_config import create_config_file, set_active_config, load_cfg
from quantizer.quantizer_POTQ import quantizer_module

from pathlib import Path

args = {
    "epochs": 50,
    "dataset": "cifar10", # choose between "tinyimagenet", "cifar10", "cifar100", "gesture"
    "name": "test", # experiment dir name 
    "frames": 16,
    "mode": "fwdPass",
    "channels": 2,
    "polarity": "both",
    "seed": 42,
    "verbose": 2,
    "dataset_path": "./data", ## Choose your dataset path
    "cache_dataset": "./"
}

exp_dir = Path(args["name"])
exp_dir.mkdir(parents=True, exist_ok=True)


cfg_path = create_config_file(exp_dir.name, overrides=args)
set_active_config(cfg_path)
cfg = load_cfg(force=True)

required_dirs = [
    "Model", "database", "log_folder",
    "algorithm_logs", "dashboard", "dashboard/model", "symbolic"
]

for folder in required_dirs:
    (exp_dir / folder).mkdir(exist_ok=True)

X_train, Y_train, X_test, Y_test, n_classes = get_datasets(cfg.dataset)

params = {'num_neurons': 32, 
            'unit_c1': 2, 
            'unit_c2': 3, 
            'new_fc_1': 59,
            'dr_f': 0.077, 
            'learning_rate': 0.001, 
            'batch_size': 30, 
            'optimizer': 'Adamax', 
            'activation': 'relu', 
            'data_augmentation': True, 
            'reg_l2': False, 
            'skip_connection': False, 
            }

nn = neural_network(X_train, Y_train, X_test, Y_test, n_classes, 
                    params['reg_l2'], params['data_augmentation'], params['skip_connection'])
nn.build_network(params)


# train and test the model
scoreNN, history, model = nn.training(params)
print(f"Test loss: {scoreNN[0]}")
print(f"Test accuracy: {scoreNN[1]}")

quantizer = quantizer_module(opt=params["optimizer"])
quantized_model = quantizer.quantizer_function(model)
score = quantizer.evaluate_quantized_model(X_test, Y_test)
print(f"Test loss quantizer: {score[0]}")
print(f"Test accuracy quantizer: {score[1]}")