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
            
# generate_params.py
from pathlib import Path

def generate_params_file(output_path: str = "params.txt"):
    """
    Generate a params.txt file for job arrays, where each line is a parameter combination
    starting from 'gesture'. Ignores prefixes and IDs.
    """
    datasets = ["roigesture_matrix"] # "roigesture_matrix", "roigesture_coords", 
    params_th = [1000000, 10000000000]  # 1M and 100G params, for example
    seeds = [42, 7, 96]  # Example seeds

    # configuration definitions
    configs = [
        # ("depth",   [(4, 4), (8, 8), (16, 16), (32, 32), (64, 64)]),
        ("fwdPass", [10000, 100000, 200000]), 
        #("hybrid",  [(16, 4), (16, 8), (32, 4), (32, 8), (64, 4), (64, 8), (64, 16)]),
    ]

    lines = []
    for dataset, (mode, params_list), seed in product(datasets, configs, seeds):
            for p in params_list:
                line = f"{dataset},{mode},{p},{seed}"
                lines.append(line)

    # write the file
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✅ File '{output_path}' generato con {len(lines)} combinazioni.")

def generate_roi_k_params_file(output_path: str = "params_roi_k.txt",
                               cache_output_path: str = "params_roi_k_cache.txt"):
    """
    Job array for the K-sub-window ROI variants (fixed delta_t framing only).

    params_roi_k.txt       -> one line per job:  dataset,mode,delta_t,roi_k,seed
    params_roi_k_cache.txt -> one line per (delta_t, roi_k): delta_t,roi_k
                              (the ROI cache is shared by both variants and all
                              seeds, so it is built once per pair before the array)
    roi_k is 10, except at 5 ms where 10 sub-windows would be 0.5 ms (< 1 ms ROI
    resolution): there roi_k = 5.
    """
    datasets = ["roigesture_conv1d", "roigesture_channels"]
    seeds = [42, 7, 96]
    dt_k = [(5000, 5), (10000, 10), (100000, 10), (200000, 10)]  # (delta_t us, roi_k)

    lines = [f"{dataset},fwdPass,{dt},{k},{seed}"
             for dataset, (dt, k), seed in product(datasets, dt_k, seeds)]
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(cache_output_path).write_text("\n".join(f"{dt},{k}" for dt, k in dt_k) + "\n", encoding="utf-8")
    print(f"✅ '{output_path}': {len(lines)} jobs; '{cache_output_path}': {len(dt_k)} cache builds.")


def generate_roi_all_params_file(output_path: str = "params_roi_all.txt",
                                 cache_output_path: str = "params_roi_all_cache.txt"):
    """
    Job array re-running every ROI variant at 5, 10, 100 and 200 ms (fixed
    delta_t framing on train and test).

    params_roi_all.txt       -> one line per job:  dataset,mode,delta_t,roi_k,seed
                                (roi_k = 0 for the variants without sub-windows)
    params_roi_all_cache.txt -> one line per cache: delta_t,roi_k
                                (roi_k = 0: plain delta_t ROI cache, shared by
                                coords / matrix / matrix_fc and all seeds)
    roi_k as in generate_roi_k_params_file: 5 at 5 ms, 10 otherwise.
    """
    plain_datasets = ["roigesture_coords", "roigesture_matrix", "roigesture_matrix_fc"]
    roi_k_datasets = ["roigesture_conv1d", "roigesture_channels"]
    seeds = [42, 7, 96]
    dt_k = [(5000, 5), (10000, 10), (100000, 10), (200000, 10)]  # (delta_t us, roi_k)

    jobs = [(dataset, dt, 0) for dataset, (dt, _) in product(plain_datasets, dt_k)]
    jobs += [(dataset, dt, k) for dataset, (dt, k) in product(roi_k_datasets, dt_k)]
    lines = [f"{dataset},fwdPass,{dt},{k},{seed}" for (dataset, dt, k), seed in product(jobs, seeds)]
    caches = [(dt, 0) for dt, _ in dt_k] + dt_k
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(cache_output_path).write_text("\n".join(f"{dt},{k}" for dt, k in caches) + "\n", encoding="utf-8")
    print(f"✅ '{output_path}': {len(lines)} jobs; '{cache_output_path}': {len(caches)} cache builds.")


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "roi_k":
        generate_roi_k_params_file()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "roi_all":
        generate_roi_all_params_file()
        return
    job_configs = generate_jobs()
    # save_job_configs_to_file(job_configs, "params.txt")
    generate_params_file()

if __name__ == "__main__":
    main()