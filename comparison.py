import itertools
import json
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

def compare(logliks: Tuple[dict, dict]):

    _, num_datasets = _check_consistency(logliks)
    counter = len(logliks[0])
    
    for d in range(num_datasets):
        if np.exp(logliks[0][str(d)][1]) < np.exp(logliks[1][str(d)][1]):
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
        context_size = logliks[0][str(d)][0]
        for m in range(1, num_models):
            assert context_size == logliks[m][str(d)][0]
    
    return num_models, num_datasets


def _process_data(logliks: List[dict], avg_context: bool = True, log: bool = False):

    num_models, num_datasets = _check_consistency(logliks)
    
    datasets = {}
    for d in range(num_datasets):
        for m in range(0, num_models):
            if m == 0:
                if log:
                    datasets[d] = [logliks[m][str(d)][0], logliks[m][str(d)][1]]
                else:
                    datasets[d] = [logliks[m][str(d)][0], np.exp(logliks[m][str(d)][1])]
            else:
                if log:
                    datasets[d].append(logliks[m][str(d)][1])
                else:
                    datasets[d].append(np.exp(logliks[m][str(d)][1]))
    
    if avg_context:
        # Averaging over context sizes
        data_by_context = {}
        for dataset in datasets.values():
            if dataset[0] not in data_by_context:
                data_by_context[dataset[0]] = np.array(dataset[1:])
            else:
                data_by_context[dataset[0]] = np.row_stack([data_by_context[dataset[0]], np.array(dataset[1:])])
        for k, v in data_by_context.items():
            data_by_context[k] = np.mean(v, axis=0) if v.ndim>1 else v

        # Sorting by context size
        out = dict(sorted(data_by_context.items(), key=lambda item: item[0]))
    else:
        # Sorting by context size (and by loglik of first model)
        out = dict(sorted(datasets.items(), key=lambda item: (item[1][0], item[1][1])))

    return out


def plot_hist_comparison_by_context(logliks: List[dict], labels: List[str], filename: str, log: bool = False):

    assert len(logliks) == len(labels)

    num_models = len(logliks)
    data = _process_data(logliks, True, log)
    pos = np.arange(0, len(data))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:]
    colors.extend(colors)
    width = 0.85/num_models

    plt.figure(figsize=(16,8))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Context set size", fontsize=20, labelpad=10)
    plt.ylabel("Likelihood", fontsize=20, labelpad=10)

    for x, values in zip(pos, data.values()):
        for i, value in enumerate(values):
            position = x + (width*(1-num_models)/2) + i*width
            plt.bar(position, value, width, color=colors[i], label=labels[i])

    # Adding legend
    ax = plt.gca()
    h, l = ax.get_legend_handles_labels()
    hprime = []
    lprime = []
    for m in range(num_models):
        idx = l.index(labels[m])
        hprime.append(h[idx])
        lprime.append(l[idx])
    # ax.legend(hprime, lprime)
    leg = ax.legend(hprime, lprime, facecolor="#eeeeee", edgecolor="#ffffff", framealpha=0.85, loc="upper left", labelspacing=0.25, fontsize=14)
    leg.get_frame().set_linewidth(0)

    plt.xlim([-1, 31])
    plt.xticks(pos)

    # For poster consistency:
    ax.set_axisbelow(True)  # Show grid lines below other elements.
    ax.grid(which="major", c="#c0c0c0", alpha=0.5, lw=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_lw(1)
    ax.spines["left"].set_lw(1)
    ax.xaxis.set_ticks_position("bottom")
    ax.xaxis.set_tick_params(width=1)
    ax.yaxis.set_ticks_position("left")
    ax.yaxis.set_tick_params(width=1)

    plt.tight_layout()
    plt.savefig(f"images/{filename}.png", dpi=600)
    plt.close()


# def plot_hist_comparison_by_dataset(logliks: List[dict], labels: List[str], filename: str, log: bool = False):

#     assert len(logliks) == len(labels)

#     num_models = len(logliks)
#     num_context_points = []
#     data = _process_data(logliks, False, log)
#     pos = np.arange(0, len(data), 1)
#     colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:]

#     plt.figure(figsize=(16,8))
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.xlabel("Dataset index", fontsize=20, labelpad=10)
#     plt.ylabel("Likelihood", fontsize=20, labelpad=10)

#     bars = []
#     for x, values in zip(pos, data.values()):
#         num_context_points.append(values[0])
#         for i, (h, c, l) in enumerate(sorted(zip(values[1:], colors, range(len(labels))))):
#             bar = plt.bar(x, h, color=c, zorder=5-i, label=labels[l])
#             if i == num_models - 1:
#                 bars.append(bar)

#     # Adding legend
#     ax = plt.gca()
#     h, l = ax.get_legend_handles_labels()
#     hprime = []
#     lprime = []
#     for m in range(num_models):
#         idx = l.index(labels[m])
#         hprime.append(h[idx])
#         lprime.append(l[idx])
#     # ax.legend(hprime, lprime)
#     leg = ax.legend(hprime, lprime, facecolor="#eeeeee", edgecolor="#ffffff", framealpha=0.85, loc="upper left", labelspacing=0.25, fontsize=14)
#     leg.get_frame().set_linewidth(0)

#     for bar, cont in zip(bars, num_context_points):
#         ax.bar_label(bar, labels=[cont], fontsize=8, fontweight="bold")

#     plt.xlim([-1, 100])

#     # For poster consistency:
#     ax.set_axisbelow(True)  # Show grid lines below other elements.
#     ax.grid(which="major", c="#c0c0c0", alpha=0.5, lw=1)
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.spines["bottom"].set_lw(1)
#     ax.spines["left"].set_lw(1)
#     ax.xaxis.set_ticks_position("bottom")
#     ax.xaxis.set_tick_params(width=1)
#     ax.yaxis.set_ticks_position("left")
#     ax.yaxis.set_tick_params(width=1)

#     plt.tight_layout()
#     plt.savefig(f"images/{filename}.png", dpi=400)
#     plt.close()


def plot_line_comparison_by_context(logliks: List[dict], labels: List[str], filename: str, log: bool = False):

    assert len(logliks) == len(labels)

    num_models = len(logliks)
    data = _process_data(logliks, True, log)
    x = list(data.keys())
    # colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:]

    plt.figure(figsize=(16,8))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Context set size", fontsize=20, labelpad=10)
    plt.ylabel("Likelihood", fontsize=20, labelpad=10)

    lists = {}
    for label in labels:
        lists[label] = []
    for value in data.values():
        for idx, label in enumerate(labels):
            lists[label].append(value[idx])

    for label in lists.keys():
        plt.plot(x, lists[label], label=label, linewidth=2, marker="x", markersize=8, markeredgewidth=2)
        # linestyle="dashed", 

    # Adding legend
    ax = plt.gca()
    leg = ax.legend(facecolor="#eeeeee", edgecolor="#ffffff", framealpha=0.85, loc="upper left", labelspacing=0.25, fontsize=14)
    leg.get_frame().set_linewidth(0)

    plt.xlim([x[0]-1, x[-1]+1])
    plt.xticks(x)

    # For poster consistency:
    ax.set_axisbelow(True)  # Show grid lines below other elements.
    ax.grid(which="major", c="#c0c0c0", alpha=0.5, lw=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_lw(1)
    ax.spines["left"].set_lw(1)
    ax.xaxis.set_ticks_position("bottom")
    ax.xaxis.set_tick_params(width=1)
    ax.yaxis.set_ticks_position("left")
    ax.yaxis.set_tick_params(width=1)

    plt.tight_layout()
    plt.savefig(f"images/{filename}.png", dpi=500)
    plt.close()


if __name__ == "__main__":

    # DATASETS = ["noised_sawtooth", "noised_square_wave"]
    Y_DIMS = [3, 4, 5, 6]
    NUMS_AR_SAMPLES = [100, 1000]
    ARCHS = ["s64_n6_k5", "s70_n10_k5", "s80_n12_k5"]
    MODELS = ["joint", "split"]

    # # Plot architecture comparisons
    # for dataset, y_dim, num_ar_samples in itertools.product(DATASETS, Y_DIMS, NUMS_AR_SAMPLES):
    #     with open(f"_experiments/{dataset}/x1_y{y_dim}/convcnp/unet/s64_n6_k5/500/eval/{num_ar_samples}/logliks.json", "r") as f:
    #         small = json.load(f)
    #     with open(f"_experiments/{dataset}/x1_y{y_dim}/convcnp/unet/s70_n10_k5/500/eval/{num_ar_samples}/logliks.json", "r") as f:
    #         medium = json.load(f)
    #     with open(f"_experiments/{dataset}/x1_y{y_dim}/convcnp/unet/s80_n12_k5/500/eval/{num_ar_samples}/logliks.json", "r") as f:
    #         large = json.load(f)
    #     plot_hist_comparison_by_context([small, medium, large], ["s64_n6_k5", "s70_n10_k5", "s80_n12_k5"], f"arch_comp/{dataset}_x1_y{y_dim}_{num_ar_samples}")

    # # Plot y_dim comparisons
    # for dataset, arch, num_ar_samples in itertools.product(DATASETS, ARCHS, NUMS_AR_SAMPLES):
    #     with open(f"_experiments/{dataset}/x1_y3/convcnp/unet/{arch}/500/eval/{num_ar_samples}/logliks.json", "r") as f:
    #         l3 = json.load(f)
    #     with open(f"_experiments/{dataset}/x1_y4/convcnp/unet/{arch}/500/eval/{num_ar_samples}/logliks.json", "r") as f:
    #         l4 = json.load(f)
    #     with open(f"_experiments/{dataset}/x1_y5/convcnp/unet/{arch}/500/eval/{num_ar_samples}/logliks.json", "r") as f:
    #         l5 = json.load(f)
    #     with open(f"_experiments/{dataset}/x1_y6/convcnp/unet/{arch}/500/eval/{num_ar_samples}/logliks.json", "r") as f:
    #         l6 = json.load(f)
    #     plot_hist_comparison_by_context([l3, l4, l5, l6], ["3 layers", "4 layers", "5 layers", "6 layers"], f"ydim_comp/{dataset}_{arch}_{num_ar_samples}")

    # # Plot ar_samples comparison
    # for dataset, arch, y_dim in itertools.product(DATASETS, ARCHS, Y_DIMS):
    #     with open(f"_experiments/{dataset}/x1_y{y_dim}/convcnp/unet/{arch}/500/eval/100/logliks.json", "r") as f:
    #         s100 = json.load(f)
    #     with open(f"_experiments/{dataset}/x1_y{y_dim}/convcnp/unet/{arch}/500/eval/1000/logliks.json", "r") as f:
    #         s1000 = json.load(f)
    #     plot_hist_comparison_by_context([s100, s1000], ["100 AR samples", "1000 AR samples"], f"ar_samp_comp/{dataset}_x1_y{y_dim}_{arch}")

    # # Plot split vs joint (100 samples)
    # for arch, y_dim in itertools.product(ARCHS, Y_DIMS):
    #     with open(f"_experiments/noised_sawtooth/joint/{y_dim}_layers/convcnp/unet/{arch}/500_epochs/eval/100/logliks.json", "r") as f:
    #         joint = json.load(f)
    #     with open(f"_experiments/noised_sawtooth/split/{y_dim}_layers/convcnp/unet/{arch}/500_epochs/eval/100/logliks.json", "r") as f:
    #         split = json.load(f)
    #     plot_hist_comparison_by_context([split, joint], ["Split", "Joint"], f"split_vs_joint/{y_dim}_layers_{arch}")

    # # Plot 100 vs 1000 for available split data (i.e. just 3 and 4 layers). Hopefully these are similar.
    # for arch, y_dim, model in itertools.product(ARCHS, [3, 4], MODELS):
    #     with open(f"_experiments/noised_sawtooth/{model}/{y_dim}_layers/convcnp/unet/{arch}/500_epochs/eval/100/logliks.json", "r") as f:
    #         s100 = json.load(f)
    #     with open(f"_experiments/noised_sawtooth/{model}/{y_dim}_layers/convcnp/unet/{arch}/500_epochs/eval/1000/logliks.json", "r") as f:
    #         s1000 = json.load(f)
    #     plot_hist_comparison_by_context([s100, s1000], ["100 AR samples", "1000 AR samples"], f"ar_samp_comp/{y_dim}_layers_{arch}_{model}")

    # # Re-plot y_dim comparisons
    # for arch, model in itertools.product(ARCHS, MODELS):
    #     with open(f"_experiments/noised_sawtooth/{model}/3_layers/convcnp/unet/{arch}/500_epochs/eval/100/logliks.json", "r") as f:
    #         l3 = json.load(f)
    #     with open(f"_experiments/noised_sawtooth/{model}/4_layers/convcnp/unet/{arch}/500_epochs/eval/100/logliks.json", "r") as f:
    #         l4 = json.load(f)
    #     with open(f"_experiments/noised_sawtooth/{model}/5_layers/convcnp/unet/{arch}/500_epochs/eval/100/logliks.json", "r") as f:
    #         l5 = json.load(f)
    #     with open(f"_experiments/noised_sawtooth/{model}/6_layers/convcnp/unet/{arch}/500_epochs/eval/100/logliks.json", "r") as f:
    #         l6 = json.load(f)
    #     plot_hist_comparison_by_context([l3, l4, l5, l6], ["3 layers", "4 layers", "5 layers", "6 layers"], f"ydim_comp/{arch}_{model}")

    # # Re-plot architecture comparisons
    # for y_dim, model in itertools.product(Y_DIMS, MODELS):
    #     with open(f"_experiments/noised_sawtooth/{model}/{y_dim}_layers/convcnp/unet/s64_n6_k5/500_epochs/eval/100/logliks.json", "r") as f:
    #         small = json.load(f)
    #     with open(f"_experiments/noised_sawtooth/{model}/{y_dim}_layers/convcnp/unet/s70_n10_k5/500_epochs/eval/100/logliks.json", "r") as f:
    #         medium = json.load(f)
    #     with open(f"_experiments/noised_sawtooth/{model}/{y_dim}_layers/convcnp/unet/s80_n12_k5/500_epochs/eval/100/logliks.json", "r") as f:
    #         large = json.load(f)
    #     plot_hist_comparison_by_context([small, medium, large], ["s64_n6_k5", "s70_n10_k5", "s80_n12_k5"], f"arch_comp/{y_dim}_layers_{model}")

    # # Compare 1 and 2 stride in OG model and others
    # with open(f"_experiments/sawtooth/x1_y1_stride1/convcnp/unet/loglik/logliks.json", "r") as f:
    #     stride1 = json.load(f)
    # with open(f"_experiments/sawtooth/x1_y1_stride2/convcnp/unet/loglik/logliks.json", "r") as f:
    #     stride2 = json.load(f)
    # plot_hist_comparison_by_context([stride1, stride2], ["Stride 1", "Stride 2"], f"stride_comp/OG_model")
    # for arch, y_dim in itertools.product(ARCHS, Y_DIMS):
    #     with open(f"_experiments_backup/noised_sawtooth/x1_y{y_dim}/convcnp/unet/{arch}/500/eval/1000/logliks.json", "r") as f:
    #         stride1 = json.load(f)
    #     with open(f"_experiments/noised_sawtooth/joint/{y_dim}_layers/convcnp/unet/{arch}/500_epochs/eval/1000/logliks.json", "r") as f:
    #         stride2 = json.load(f)
    #     plot_hist_comparison_by_context([stride1, stride2], ["Stride 1", "Stride 2"], f"stride_comp/{y_dim}_layers_{arch}")

    # # Compare OG with 0 noise layer model
    # with open(f"_experiments/sawtooth/x1_y1_stride2/convcnp/unet/loglik/logliks.json", "r") as f:
    #     og = json.load(f)
    # with open(f"_experiments/noised_sawtooth/joint/1_layers/convcnp/unet/s64_n6_k5/500_epochs/eval/100/logliks.json", "r") as f:
    #     zero_noise = json.load(f)
    # plot_hist_comparison_by_context([og, zero_noise], ["OG", "Zero noise"], f"og_vs_zero_noise")

    # # Compare OG with basic 3 layers joint and split
    # # with open(f"_experiments/sawtooth/x1_y1_stride2/convcnp/unet/loglik/logliks.json", "r") as f:
    # #     og = json.load(f)
    # with open(f"_experiments/noised_sawtooth/joint/1_layers/convcnp/unet/s64_n6_k5/500_epochs/eval/1000/logliks.json", "r") as f:
    #     og = json.load(f)
    # with open(f"/scratch/lb953/_experiments/noised_sawtooth/joint/3_layers/convcnp/unet/s64_n6_k5/500_epochs/eval/1000/logliks.json", "r") as f:
    #     joint = json.load(f)
    # with open(f"/scratch/lb953/_experiments/noised_sawtooth/split/3_layers/convcnp/unet/s64_n6_k5/500_epochs/eval/1000/logliks.json", "r") as f:
    #     split = json.load(f)
    # plot_hist_comparison_by_context([og, joint, split], ["OG", "Joint", "Split"], f"baseline_comp/1000_ar_samples")

    # # Compare OG with AR and basic 3 layers joint and split
    # for arch in ARCHS:
    #     with open(f"_experiments/sawtooth/original_model/convcnp/unet/{arch}/500_epochs/eval/logliks_regular.json", "r") as f:
    #         og = json.load(f)
    #     with open(f"_experiments/sawtooth/original_model/convcnp/unet/{arch}/500_epochs/eval/logliks_AR.json", "r") as f:
    #         ar = json.load(f)
    #     with open(f"_experiments/noised_sawtooth/joint/3_layers/convcnp/unet/{arch}/500_epochs/eval/1000/logliks.json", "r") as f:
    #         joint = json.load(f)
    #     with open(f"_experiments/noised_sawtooth/split/3_layers/convcnp/unet/{arch}/500_epochs/eval/1000/logliks.json", "r") as f:
    #         split = json.load(f)
    #     plot_hist_comparison_by_context([og, joint, split, ar], ["OG", "Joint", "Split", "AR"], f"baseline_comp/{arch}_1000_ar_samples")

    # # Compare OG big with AR big, 3 layers joint big and 3 layers split small
    # with open(f"_experiments/sawtooth/original_model/convcnp/unet/s80_n12_k5/500_epochs/eval/logliks_regular.json", "r") as f:
    #     og = json.load(f)
    # with open(f"_experiments/sawtooth/original_model/convcnp/unet/s80_n12_k5/500_epochs/eval/logliks_AR.json", "r") as f:
    #     ar = json.load(f)
    # with open(f"_experiments/noised_sawtooth/joint/3_layers/convcnp/unet/s80_n12_k5/500_epochs/eval/1000/logliks.json", "r") as f:
    #     joint = json.load(f)
    # with open(f"_experiments/noised_sawtooth/split/3_layers/convcnp/unet/s64_n6_k5/500_epochs/eval/1000/logliks.json", "r") as f:
    #     split = json.load(f)
    # plot_hist_comparison_by_context([og, joint, split, ar], ["OG", "Joint", "Split", "AR"], f"baseline_comp/same_param_count")

    # # Compare AR architectures
    # for kind in ["regular", "AR"]:
    #     with open(f"_experiments/sawtooth/original_model/convcnp/unet/s64_n6_k5/500_epochs/eval/logliks_{kind}.json", "r") as f:
    #         small = json.load(f)
    #     with open(f"_experiments/sawtooth/original_model/convcnp/unet/s70_n10_k5/500_epochs/eval/logliks_{kind}.json", "r") as f:
    #         medium = json.load(f)
    #     with open(f"_experiments/sawtooth/original_model/convcnp/unet/s80_n12_k5/500_epochs/eval/logliks_{kind}.json", "r") as f:
    #         large = json.load(f)
    #     plot_hist_comparison_by_context([small, medium, large], ["Small", "Medium", "Large"], f"baseline_comp/{kind}_arch_comp")

    # # Compare best models
    # with open(f"_experiments/sawtooth/original_model/convcnp/unet/s64_n6_k5/500_epochs/eval/logliks_regular.json", "r") as f:
    #     og = json.load(f)
    # with open(f"_experiments/sawtooth/original_model/convcnp/unet/s64_n6_k5/500_epochs/eval/logliks_AR.json", "r") as f:
    #     ar = json.load(f)
    # with open(f"_experiments/noised_sawtooth/joint/3_layers/convcnp/unet/s64_n6_k5/500_epochs/eval/1000/logliks.json", "r") as f:
    #     joint = json.load(f)
    # with open(f"_experiments/noised_sawtooth/split/3_layers/convcnp/unet/s80_n12_k5/500_epochs/eval/1000/logliks.json", "r") as f:
    #     split = json.load(f)
    # plot_hist_comparison_by_context([og, joint, split, ar], ["OG", "Joint", "Split", "AR"], f"baseline_comp/best_models")

    # # Compare AR and no AR models
    # for arch in ARCHS:
    #     with open(f"_experiments/sawtooth/original_model/convcnp/unet/{arch}/500_epochs/eval/logliks_regular.json", "r") as f:
    #         og = json.load(f)
    #     with open(f"_experiments/sawtooth/original_model/convcnp/unet/{arch}/500_epochs/eval/logliks_AR.json", "r") as f:
    #         ar = json.load(f)
    #     plot_hist_comparison_by_context([og, ar], ["OG", "AR"], f"ar_vs_no_ar/{arch}")

    # # Compare upper-bounded noise
    # with open(f"_experiments/noised_sawtooth/joint/3_layers/convcnp/unet/s64_n6_k5/500_epochs/eval/100/logliks.json", "r") as f:
    #     l3 = json.load(f)
    # with open(f"_experiments/noised_sawtooth/joint/4_layers/convcnp/unet/s64_n6_k5/500_epochs/eval/100/logliks.json", "r") as f:
    #     l4 = json.load(f)
    # with open(f"_experiments/noised_sawtooth/joint/5_layers/convcnp/unet/s64_n6_k5/500_epochs/eval/100/logliks.json", "r") as f:
    #     l5 = json.load(f)
    # with open(f"_experiments/noised_sawtooth/joint/6_layers/convcnp/unet/s64_n6_k5/500_epochs/eval/100/logliks.json", "r") as f:
    #     l6 = json.load(f)
    # plot_hist_comparison_by_context([l3, l4, l5, l6], ["3 layers", "4 layers", "5 layers", "6 layers"], f"upper_bounded_noise_joint")


    # # Compare upper-bounded noise values
    # variances = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24]
    # data = []
    # for var in variances:
    #     with open(f"_experiments/noised_sawtooth/split/3_layers/convcnp/unet/s64_n6_k5/{var}_var/500_epochs/eval/100/logliks.json", "r") as f:
    #         data.append(json.load(f))
    # plot_hist_comparison_by_context(data, [f"{var} var" for var in variances], f"upper_bounded_noise_values_comp_new")

    # # Compare same vs different xt
    # with open("_experiments/noised_sawtooth/joint/3_layers/convcnp/unet/s64_n6_k5/0.02_var/diff_xt/500_epochs/eval/250/logliks.json", "r") as f:
    #     diff_xt = json.load(f)
    # with open("_experiments/noised_sawtooth/joint/3_layers/convcnp/unet/s64_n6_k5/0.02_var/same_xt/500_epochs/eval/250/logliks.json", "r") as f:
    #     same_xt = json.load(f)
    # plot_hist_comparison_by_context([same_xt, diff_xt], ["Same xt", "Diff xt"], f"same_vs_diff_xt_joint_0.02_var")

    # # Compare diff ydim with higher bound noise
    # with open("_experiments/noised_sawtooth/split/3_layers/convcnp/unet/s64_n6_k5/0.08_var/diff_xt/500_epochs/eval/250/logliks.json", "r") as f:
    #     l3 = json.load(f)
    # with open("_experiments/noised_sawtooth/split/4_layers/convcnp/unet/s64_n6_k5/0.12_var/diff_xt/500_epochs/eval/250/logliks.json", "r") as f:
    #     l4 = json.load(f)
    # # with open("_experiments/noised_sawtooth/split/5_layers/convcnp/unet/s64_n6_k5/0.08_var/diff_xt/500_epochs/eval/250/logliks.json", "r") as f:
    # #     l5 = json.load(f)
    # with open("_experiments/noised_sawtooth/split/6_layers/convcnp/unet/s64_n6_k5/0.2_var/diff_xt/500_epochs/eval/250/logliks.json", "r") as f:
    #     l6 = json.load(f)
    # plot_hist_comparison_by_context([l3, l4, l6], ["3 Layers", "4 Layers", "6 Layers"], f"uppboundnoise_increasingvars")

    # # GP results
    # with open("/scratch/lb953/_experiments/noised_gp/split/3_layers/convcnp/unet/s64_n6_k5/0.1_var/diff_xt/500_epochs/eval/250/logliks.json", "r") as f:
    #     noised_gp = json.load(f)
    # plot_hist_comparison_by_context([noised_gp], ["Noised GP"], f"noised_gp")

    # # Compare AR context
    # ar_context_sizes = [0, 6, 8, 10, 12, 14, 16, 18, 20]
    # data = []
    # for ar_context_size in ar_context_sizes:
    #     with open(f"/scratch/lb953/_experiments_pre_less_targets/noised_sawtooth/split/3_layers/convcnp/unet/s64_n6_k5/0.08_var/diff_xt/500_epochs/eval_100_targ/250/{ar_context_size}/logliks.json", "r") as f:
    #         data.append(json.load(f))
    # plot_hist_comparison_by_context(data, [f"{ar_context_size} AR context" for ar_context_size in ar_context_sizes], f"AR_context_comparison", False)

    # # Compare fewer targets
    # with open("/scratch/lb953/_experiments/noised_sawtooth/split/3_layers/convcnp/unet/s64_n6_k5/0.08_var/diff_xt/500_epochs/eval/250/0/logliks.json", "r") as f:
    #     few_targ = json.load(f)
    # with open("/scratch/lb953/_experiments_pre_less_targets/noised_sawtooth/split/3_layers/convcnp/unet/s64_n6_k5/0.08_var/diff_xt/500_epochs/eval/250/0/logliks.json", "r") as f:
    #     reg_targ = json.load(f)
    # plot_hist_comparison_by_context([few_targ, reg_targ], ["Variable targ (0-50)", "100 targets"], f"few_vs_many_targets_diff_targ_sizes")

    # # More GP results
    # data = []
    # for ar_samples in [100, 1000, 10000]:
    #     with open(f"/scratch/lb953/_experiments/noised_gp/joint/3_layers/convcnp/unet/s64_n6_k5/0.02_var/diff_xt/500_epochs/eval/{ar_samples}/0/logliks.json", "r") as f:
    #         data.append(json.load(f))
    # with open("/scratch/lb953/_experiments/noised_gp/split/3_layers/convcnp/unet/s64_n6_k5/0.02_var/diff_xt/500_epochs/eval/300/0/logliks.json", "r") as f:
    #     data.append(json.load(f))
    # plot_hist_comparison_by_context(data, ["Joint 100", "Joint 1000", "Joint 10000", "Split 300"], f"noised_gp")

    # # Compare GP with baseline
    # with open(f"/scratch/lb953/_experiments/noised_gp/joint/3_layers/convcnp/unet/s64_n6_k5/0.02_var/diff_xt/500_epochs/eval/10000/0/logliks.json", "r") as f:
    #     joint_10k = json.load(f)
    # with open(f"_experiments/noised_gp/joint/1_layers/convcnp/unet/s64_n6_k5/0.02_var/diff_xt/500_epochs/eval/100/0/logliks.json", "r") as f:
    #     baseline = json.load(f)
    # plot_hist_comparison_by_context([baseline, joint_10k], ["Baseline", "Joint 10000"], f"noised_gp_baseline")

    # # More AR context comparison
    # ar_context_sizes = [0, 10, 20, 40, 60]
    # data = []
    # for ar_context_size in ar_context_sizes:
    #     with open(f"/scratch/lb953/_experiments_60context/noised_sawtooth/split/3_layers/convcnp/unet/s64_n6_k5/0.08_var/diff_xt/500_epochs/eval/200/{ar_context_size}/logliks.json", "r") as f:
    #         data.append(json.load(f))
    # plot_hist_comparison_by_context(data, [f"{ar_context_size} AR context" for ar_context_size in ar_context_sizes], f"AR_context_comparison_60_cont_size", False)

    # # More comparisons
    # # for data_type in ["noised_gp", "noised_sawtooth", "noised_square_wave"]:
    # for data_type in ["noised_gp"]:
    #     with open(f"/scratch/lb953/_experiments_50_targ/{data_type}/joint/3_layers/convcnp/unet/s64_n6_k5/50_targ/0.02_var/diff_xt/500_epochs/eval/1000/0/logliks.json", "r") as f:
    #         joint_002 = json.load(f)
    #     with open(f"/scratch/lb953/_experiments_50_targ/{data_type}/joint/3_layers/convcnp/unet/s64_n6_k5/50_targ/0.08_var/diff_xt/500_epochs/eval/1000/0/logliks.json", "r") as f:
    #         joint_008 = json.load(f)
    #     if data_type != "noised_square_wave":
    #         with open(f"/scratch/lb953/_experiments_50_targ/{data_type}/split/3_layers/convcnp/unet/s64_n6_k5/50_targ/0.08_var/diff_xt/500_epochs/eval/200/0/logliks.json", "r") as f:
    #             split_008 = json.load(f)
    #         plot_line_comparison_by_context([joint_002, joint_008, split_008], ["Joint, 0.02 var", "Joint, 0.08 var", "Split, 0.08 var"], f"convcnp_{data_type}_50_targets", False)
    #     else:
    #         plot_line_comparison_by_context([joint_002, joint_008], ["Joint, 0.02 var", "Joint, 0.08 var"], f"convcnp_{data_type}_50_targets", False)

    # # Even more comparisons
    # for data_type in ["noised_gp", "noised_sawtooth", "noised_square_wave"]:
    #     with open(f"/scratch/lb953/_experiments_50_targ/{data_type}/joint/3_layers/convcnp/unet/s64_n6_k5/50_targ/0.02_var/diff_xt/500_epochs/eval/1000/0/logliks.json", "r") as f:
    #         joint_002_convcnp = json.load(f)
    #     with open(f"/scratch/lb953/_experiments_50_targ/{data_type}/joint/3_layers/convgnp/unet/s64_n6_k5/50_targ/0.02_var/diff_xt/500_epochs/eval/1000/0/logliks.json", "r") as f:
    #         joint_002_convgnp = json.load(f)
    #     with open(f"/scratch/lb953/_experiments_50_targ/{data_type}/joint/3_layers/convcnp/unet/s64_n6_k5/50_targ/0.08_var/diff_xt/500_epochs/eval/1000/0/logliks.json", "r") as f:
    #         joint_008_convcnp = json.load(f)
    #     with open(f"/scratch/lb953/_experiments_50_targ/{data_type}/joint/3_layers/convgnp/unet/s64_n6_k5/50_targ/0.08_var/diff_xt/500_epochs/eval/1000/0/logliks.json", "r") as f:
    #         joint_008_convgnp = json.load(f)
    #     with open(f"/scratch/lb953/_experiments_50_targ/{data_type}/split/3_layers/convcnp/unet/s64_n6_k5/50_targ/0.08_var/diff_xt/500_epochs/eval/200/0/logliks.json", "r") as f:
    #         split_008_convcnp = json.load(f)
    #     if data_type != "noised_square_wave":
    #         with open(f"/scratch/lb953/_experiments_50_targ/{data_type}/split/3_layers/convgnp/unet/s64_n6_k5/50_targ/0.08_var/diff_xt/500_epochs/eval/200/0/logliks.json", "r") as f:
    #             split_008_convgnp = json.load(f)
    #         plot_line_comparison_by_context([joint_002_convcnp, joint_002_convgnp, joint_008_convcnp, joint_008_convgnp, split_008_convcnp, split_008_convgnp],
    #                                         ["Joint CNP, 0.02 var", "Joint GNP, 0.02 var", "Joint CNP, 0.08 var", "Joint GNP, 0.08 var", "Split CNP, 0.08 var", "Split GNP, 0.08 var"], f"{data_type}_50_targets", False)
    #     else:
    #         plot_line_comparison_by_context([joint_002_convcnp, joint_002_convgnp, joint_008_convcnp, joint_008_convgnp, split_008_convcnp],
    #                                         ["Joint CNP, 0.02 var", "Joint GNP, 0.02 var", "Joint CNP, 0.08 var", "Joint GNP, 0.08 var", "Split CNP, 0.08 var"], f"{data_type}_50_targets", False)

    # # New noise var comparisons (CONVCNP)
    # NOISE_VARS = [0.1, 0.2, 0.3, 0.4, 0.5]
    # for y_dim in Y_DIMS:
    #     data = []
    #     for noise_var in NOISE_VARS:
    #         with open(f"/scratch/lb953/_experiments_50_targ/noised_sawtooth/joint/{y_dim}_layers/convcnp/unet/s64_n6_k5/50_targ/{noise_var}_var/diff_xt/500_epochs/eval/1000/0/logliks.json", "r") as f:
    #             data.append(json.load(f))
    #     plot_line_comparison_by_context(data, [f"{var} var" for var in NOISE_VARS], f"{y_dim}_noise_var_comp_convcnp")

    # # Baseline ConvCNP comparisons
    # with open("/scratch/lb953/_experiments_baseline_maxCont30/sawtooth/convcnp/unet/500_epochs/eval/logliks.json", "r") as f:
    #     regular_30_cnp = json.load(f)
    # with open("/scratch/lb953/_experiments_baseline_maxCont30/sawtooth/convcnp/unet/500_epochs/eval_AR/logliks.json", "r") as f:
    #     ar_30_cnp = json.load(f)
    # with open("/scratch/lb953/_experiments_baseline_maxCont80/sawtooth/convcnp/unet/500_epochs/eval/logliks.json", "r") as f:
    #     regular_80_cnp = json.load(f)
    # with open("/scratch/lb953/_experiments_baseline_maxCont80/sawtooth/convcnp/unet/500_epochs/eval_AR/logliks.json", "r") as f:
    #     ar_80_cnp = json.load(f)
    # plot_line_comparison_by_context([regular_30_cnp, ar_30_cnp, regular_80_cnp, ar_80_cnp],
    #                                 ["ConvCNP (MaxCont30)", "AR ConvCNP (MaxCont30)", "ConvCNP (MaxCont80)", "AR ConvCNP (MaxCont80)"],
    #                                 "ConvCNP_baselines")

    # # Baseline ConvGNP comparisons
    # with open("/scratch/lb953/_experiments_baseline_maxCont30/sawtooth/convgnp/unet/500_epochs/eval/logliks.json", "r") as f:
    #     regular_30_gnp = json.load(f)
    # with open("/scratch/lb953/_experiments_baseline_maxCont30/sawtooth/convgnp/unet/500_epochs/eval_AR/logliks.json", "r") as f:
    #     ar_30_gnp = json.load(f)
    # with open("/scratch/lb953/_experiments_baseline_maxCont80/sawtooth/convgnp/unet/500_epochs/eval/logliks.json", "r") as f:
    #     regular_80_gnp = json.load(f)
    # with open("/scratch/lb953/_experiments_baseline_maxCont80/sawtooth/convgnp/unet/500_epochs/eval_AR/logliks.json", "r") as f:
    #     ar_80_gnp = json.load(f)
    # plot_line_comparison_by_context([regular_30_gnp, ar_30_gnp, regular_80_gnp, ar_80_gnp],
    #                                 ["ConvGNP (MaxCont30)", "AR ConvGNP (MaxCont30)", "ConvGNP (MaxCont80)", "AR ConvGNP (MaxCont80)"],
    #                                 "ConvGNP_baselines")

    # # Baseline AR ConvCNP + current (4 Aug) best ConvGNP
    # with open("/scratch/lb953/_experiments_baseline_maxCont80/sawtooth/convcnp/unet/500_epochs/eval_AR/logliks.json", "r") as f:
    #     ar_80_cnp = json.load(f)
    # with open("/scratch/lb953/_experiments_50_targ/noised_sawtooth/joint/3_layers/convgnp/unet/s64_n6_k5/50_targ/0.08_var/diff_xt/500_epochs/eval/1000/0/logliks.json", "r") as f:
    #     joint_GNP_008 = json.load(f)
    # with open("/scratch/lb953/_experiments_50_targ/noised_sawtooth/joint/3_layers/convgnp/unet/s64_n6_k5/50_targ/0.02_var/diff_xt/500_epochs/eval/1000/0/logliks.json", "r") as f:
    #     joint_GNP_002 = json.load(f)
    # plot_line_comparison_by_context([ar_80_cnp, joint_GNP_008, joint_GNP_002],
    #                                 ["AR ConvCNP (MaxCont80)", "Joint ConvGNP (0.08 var)", "Joint ConvGNP (0.02 var)"],
    #                                 "ConvCNP_baselines_with_ARDNP_ConvGNP")

    # Noise var comparison (ConvGNP)
    data = []
    # NOISE_VARS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15]
    NOISE_VARS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    for noise_var in NOISE_VARS:
        with open(f"/scratch/lb953/_experiments_50_targ_convgnp_noise_var/noised_sawtooth/joint/3_layers/convgnp/unet/s64_n6_k5/50_targ/{noise_var}_var/diff_xt/500_epochs/eval/1000/0/logliks.json", "r") as f:
            data.append(json.load(f))
    plot_line_comparison_by_context(data, [f"{var} var" for var in NOISE_VARS], "noise_var_comp_GNP_2")