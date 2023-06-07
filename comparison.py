import json
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

def compare(logliks: Tuple[dict, dict]):

    num_datasets = len(logliks[0])
    counter = len(logliks[0])
    
    for d in range(num_datasets):
        if np.exp(logliks[0][str(d)][0]) < np.exp(logliks[1][str(d)][0]):
            # print(logliks[0][str(d)], logliks[1][str(d)])
            counter -= 1
    
    return np.round(counter / num_datasets * 100, 2)


def _check_consistency(logliks: List[dict]):

    num_models = len(logliks)
    num_datasets = len(logliks[0])

    # Check that all models have the same number of datasets
    for m in range(1, num_models):
        assert num_datasets == len(logliks[m])

    # Check that corresponding datasets within each model have the same
    # number of context points
    for d in range(num_datasets):
        context_size = logliks[0][str(d)][1]
        for m in range(1, num_models):
            assert context_size == logliks[m][str(d)][1]
    
    return num_models, num_datasets


def process_data(logliks: List[dict]):

    num_models, num_datasets = _check_consistency(logliks)
    
    datasets = {}
    for d in range(num_datasets):
        for m in range(0, num_models):
            if m == 0:
                datasets[d] = [logliks[m][str(d)][1], np.exp(logliks[m][str(d)][0])]
            else:
                datasets[d].append(np.exp(logliks[m][str(d)][0]))

    # Sorting by context size (and by loglik of first model)
    datasets = dict(sorted(datasets.items(), key=lambda item: (item[1][0], item[1][1])))

    return datasets


def plot_hist_comparison(logliks: List[dict], labels: List[str], filename: str):

    assert len(logliks) == len(labels)

    num_models = len(logliks)
    num_context_points = []
    data = process_data(logliks)
    pos = np.arange(0, len(data), 1)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:]

    plt.figure(figsize=(16,8))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Dataset index", fontsize=20, labelpad=20)
    plt.ylabel("Likelihood", fontsize=20, labelpad=20)

    bars = []
    for x, values in zip(pos, data.values()):
        num_context_points.append(values[0])
        for i, (h, c, l) in enumerate(sorted(zip(values[1:], colors, range(len(labels))))):
            bar = plt.bar(x, h, color=c, zorder=5-i, label=labels[l])
            if i == num_models - 1:
                bars.append(bar)

    # Adding legend
    ax = plt.gca()
    h, l = ax.get_legend_handles_labels()
    hprime = []
    lprime = []
    for m in range(num_models):
        idx = l.index(labels[m])
        hprime.append(h[idx])
        lprime.append(l[idx])
    ax.legend(hprime, lprime)

    for bar, cont in zip(bars, num_context_points):
        ax.bar_label(bar, labels=[cont], fontsize=6, fontweight="bold")

    plt.grid()
    plt.tight_layout()
    plt.savefig(f"images/{filename}.png", dpi=600)
    plt.close()


if __name__ == "__main__":

    with open("logliks_convcnp.json", "r") as f:
        convcnp = json.load(f)
    
    with open("logliks_new_100_samples.json", "r") as f:
        new_100_samples = json.load(f)

    with open("logliks_new_1000_samples.json", "r") as f:
        new_1000_samples = json.load(f)

    plot_hist_comparison([convcnp, new_100_samples, new_1000_samples], ["Baseline", "Noised (100 samples)", "Noised (1000 samples)"], "loglik_comparison")
    print(compare((new_100_samples, convcnp)))
    print(compare((new_1000_samples, convcnp)))
    print(compare((new_1000_samples, new_100_samples)))
