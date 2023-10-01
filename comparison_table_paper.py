import itertools
import json

import numpy as np
import scipy as sc

import experiment as exp
import lab as B


def print_mean_err(dataset, file):

    data = json.load(file)
    logliks = [datum[1] for datum in data.values()]
    likelihoods = np.exp(logliks)
    mean = np.mean(likelihoods)
    err = 1.96 * B.std(likelihoods) / B.sqrt(B.length(likelihoods))
    print(f"{dataset}: {mean:.2f} ± {err:.2f}")
    

if __name__ == "__main__":

    print("DANP:")
    for (dataset, depth, var) in [("noised_sawtooth", 4, 0.02), ("noised_square_wave", 4, 0.06), ("noised_gp", 3, 0.08)]:
        SAMPLES = 5000
        MIXED_MODELS = True
        if MIXED_MODELS:
            with open(f"/scratch/lb953/mixed_model_{dataset[7:]}_logliks.json") as f:
                print_mean_err(dataset, f)
        else:
            with open(f"/scratch/lb953/best_models/{dataset}/joint/{depth}_layers/convgnp/unet/s64_n6_k5/50_targ/{var}_var/diff_xt/500_epochs/eval/{SAMPLES}/0/logliks.json", "r") as f:
                print_mean_err(dataset, f)

    print("ConvCNP:")
    for dataset in ["sawtooth", "square_wave", "eq"]:
        with open(f"/scratch/lb953/_experiments_baseline_maxCont80/{dataset}/convcnp/unet/500_epochs/eval/logliks.json", "r") as f:
            print_mean_err(dataset, f)

    print("ConvGNP:")
    for dataset in ["sawtooth", "square_wave", "eq"]:
        with open(f"/scratch/lb953/_experiments_baseline_maxCont80/{dataset}/convgnp/unet/500_epochs/eval/logliks.json", "r") as f:
            print_mean_err(dataset, f)

    print("AR ConvCNP:")
    for dataset in ["sawtooth", "square_wave", "eq"]:
        with open(f"/scratch/lb953/_experiments_baseline_maxCont80/{dataset}/convcnp/unet/500_epochs/eval_AR/logliks.json", "r") as f:
            print_mean_err(dataset, f)
