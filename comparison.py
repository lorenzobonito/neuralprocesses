import json
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

def compare(logliks: Tuple[dict, dict]):

    num_datasets = len(logliks[0])
    counter = len(logliks[0])
    
    for d in range(num_datasets):
        if logliks[0][str(d)][0] < logliks[1][str(d)][0]:
            # print(logliks[0][str(d)], logliks[1][str(d)])
            counter -= 1
    
    return np.round(counter / num_datasets * 100, 2)


def plot_hist_comparison(logliks: List[dict], labels: List[str], filename: str):

    assert len(logliks) == len(labels)

    num_models = len(logliks)
    pos = np.arange(1, len(logliks[0]) + 1, 1)

    data = []
    num_context_points = [loglik[1] for loglik in logliks[0].values()]
    for m in range(num_models):
        data.append(np.exp([loglik[0] for loglik in logliks[m].values()]))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:num_models]

    plt.figure(figsize=(16,8))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Dataset index", fontsize=20, labelpad=20)
    plt.ylabel("Likelihood", fontsize=20, labelpad=20)

    bars = []
    for x, *values in zip(pos, *data):
        for i, (h, c, l) in enumerate(sorted(zip(values, colors, range(len(labels))))):                
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
    
    with open("logliks_noised_convcnp_100_samples.json", "r") as f:
        regular_100_samples = json.load(f)  
    
    with open("logliks_noised_convcnp_1000_samples.json", "r") as f:
        regular_1000_samples = json.load(f)

    with open("logliks_layer1_onlyOGcontext_100_samples.json", "r") as f:
        layer1_OGcontext_100_samples = json.load(f)  
    
    with open("logliks_layer1_onlyOGcontext_1000_samples.json", "r") as f:
        layer1_OGcontext_1000_samples = json.load(f)

    plot_hist_comparison([convcnp, regular_100_samples, regular_1000_samples], ["Baseline", "Noised (100)", "Noised (1000)"], "loglik_comparison_regular")
    plot_hist_comparison([convcnp, layer1_OGcontext_100_samples, layer1_OGcontext_1000_samples], ["Baseline", "Noised (100)", "Noised (1000)"], "loglik_comparison_level1_OGcont")
    plot_hist_comparison([convcnp, regular_100_samples, layer1_OGcontext_100_samples], ["Baseline", "Regular", "Layer 1 OG context"], "loglik_comparison_noised_100")
    plot_hist_comparison([convcnp, regular_1000_samples, layer1_OGcontext_1000_samples], ["Baseline", "Regular", "Layer 1 OG context"], "loglik_comparison_noised_1000")

    # Printing percentage of datasets in which model 1 performs better than model 2
    print(compare((regular_100_samples, convcnp)))
    print(compare((layer1_OGcontext_100_samples, convcnp)))
    print(compare((regular_1000_samples, convcnp)))
    print(compare((layer1_OGcontext_1000_samples, convcnp)))
    print(compare((regular_100_samples, layer1_OGcontext_100_samples)))
    print(compare((regular_1000_samples, layer1_OGcontext_1000_samples)))