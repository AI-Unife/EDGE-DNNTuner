import subprocess
from itertools import product

datasets_cifar = ['beans']
optimizers = ['filtered', 'RS_ruled', 'basic', 'standard', 'RS']
seeds = [42, 123, 96, 7, 84]
weights = [0.5]  # Example weights for flops and latency in the combined score


def generate_jobs():
    job_configs = []

    # CIFAR - flops
    for w, optimizer, seed, dataset in product(weights, optimizers, seeds, datasets_cifar,):
        job_configs.append({
            "data_name": dataset,
            "opt": optimizer,
            "seed": seed,
            "weight": w
        })
    return job_configs



def save_job_configs_to_file(job_configs, filename="params.txt"):
    # Sort by dataset
    job_configs = sorted(job_configs, key=lambda x: x["data_name"])

    with open(filename, "w") as f:
        for config in job_configs:
            line = ",".join(str(config[k]) for k in ["data_name", "opt", "seed", "weight"])
            f.write(line + ",flops_module\n")
            
# generate_params.py
from pathlib import Path

def generate_params_file(output_path: str = "params.txt"):
    """
    Generate a params.txt file for job arrays, where each line is a parameter combination
    starting from 'gesture'. Ignores prefixes and IDs.
    """
    datasets = ["roigesture_matrix", "roigesture_coords"] # "roigesture_matrix", "roigesture_coords", 
    params_th = [1000000, 10000000000]  # 1M and 100G params, for example
    

    # configuration definitions
    configs = [
        ("depth",   [(4, 4), (8, 8), (16, 16), (32, 32), (64, 64)]),
        ("fwdPass", [(4, 2), (8, 2), (16, 2), (32, 2), (64, 2)]),
        ("hybrid",  [(16, 4), (16, 8), (32, 4), (32, 8), (64, 4), (64, 8), (64, 16)]),
    ]

    lines = []
    for dataset in datasets:
        for param in params_th:
            for mode, params_list in configs:
                for p in params_list:
                    line = f"{dataset},{mode},{p[0]},{p[1]},{param},flops_module"
                    lines.append(line)

    # write the file
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✅ File '{output_path}' generato con {len(lines)} combinazioni.")

def main():
    job_configs = generate_jobs()
    save_job_configs_to_file(job_configs, "params_beans.txt")
    # generate_params_file()

if __name__ == "__main__":
    main()
