import subprocess
from itertools import product

datasets_cifar = ['cifar100', 'tinyimagenet']
optimizers = ['filtered', 'RS_ruled']
seeds = [42, 7, 96] #123, 84, 
weights = [0.3, 0.5, 0.7]  # Example weights for flops and latency in the combined score
modules = ['flops_module', 'hardware_module']  # List of modules to include in the job configurations


def generate_jobs():
    job_configs = []

    # CIFAR - flops
    for module, optimizer, omega, seed, dataset in product(modules, optimizers, weights, seeds, datasets_cifar,):
        job_configs.append({
            "data_name": dataset,
            "opt": optimizer,
            "seed": seed,
            "module": module,
            # "weight": w, 
            "omega": omega, 
        })
    return job_configs



def save_job_configs_to_file(job_configs, filename="params.txt"):
    # Sort by dataset
    job_configs = sorted(job_configs, key=lambda x: x["data_name"])

    with open(filename, "w") as f:
        for config in job_configs:
            line = ",".join(str(config[k]) for k in ["data_name", "opt", "seed", "module", "omega"])
            f.write(line + "\n")
            
def main():
    job_configs = generate_jobs()
    save_job_configs_to_file(job_configs, "params.txt")

if __name__ == "__main__":
    main()
