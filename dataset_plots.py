import itertools
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import torch
import lab as B
import neuralprocesses.torch as nps
from neuralprocesses.data.noised_sawtooth import NoisedSawtoothGenerator
from neuralprocesses.data.noised_square_wave import NoisedSquareWaveGenerator
from neuralprocesses.data.noised_gp import NoisedGPGenerator
from neuralprocesses.dist.uniform import UniformDiscrete, UniformContinuous
from neuralprocesses.aggregate import Aggregate, AggregateInput

X_RANGE_CONTEXT = (-2, 2)
X_RANGE_TARGET = (-2, 2)


def dataset_plot(batch):

    NUM_TARG = 30

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:]

    plt.figure(figsize=(10,6))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    zipped_t0 = zip(batch["xt"][0][0].squeeze().cpu(), batch["yt"][0].squeeze().cpu())
    t0_subset = list(zipped_t0)[:NUM_TARG]
    zipped_t0 = zip(batch["xt"][0][0].squeeze().cpu(), batch["yt"][0].squeeze().cpu())
    sorted_zip = sorted(zipped_t0, key = lambda x: x[0])
    zipped_t1 = zip(batch["xt"][1][0].squeeze().cpu(), batch["yt"][1].squeeze().cpu())
    t1_subset = list(zipped_t1)[:NUM_TARG]
    zipped_t2 = zip(batch["xt"][2][0].squeeze().cpu(), batch["yt"][2].squeeze().cpu())
    t2_subset = list(zipped_t2)[:NUM_TARG]
    plt.plot([d[0] for d in sorted_zip], [d[1] for d in sorted_zip], label="True function", linewidth=2, color="#4BA6FB")
    plt.scatter([d[0] for d in t0_subset], [d[1] for d in t0_subset], marker="x", s=30, linewidth=2, label="Original targets", color=colors[1], zorder=2)
    plt.scatter(batch["contexts"][0][0].squeeze().cpu(), batch["contexts"][0][1].squeeze().cpu(), marker="o", s=20, linewidth=2, label="Original context", color="blue", zorder=2)
    plt.scatter([d[0] for d in t1_subset], [d[1] for d in t1_subset], marker="^", s=20, linewidth=2, label="Auxiliary context #1", color=colors[2], zorder=2)
    plt.scatter([d[0] for d in t2_subset], [d[1] for d in t2_subset], marker="^", s=20, linewidth=2, label="Auxiliary context #2", color=colors[3], zorder=2)

    plt.gca().set_ylim(top=1.6)

    # Adding legend
    ax = plt.gca()
    leg = ax.legend(facecolor="#eeeeee", edgecolor="#ffffff", framealpha=0.85, loc="upper right", labelspacing=0.25, fontsize=14)
    leg.get_frame().set_linewidth(0)

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
    plt.savefig(f"images/square_wave_dataset.png", dpi=400)
    plt.close()



def get_batches(num_context: int, num_batches: int, gen_type: str, target_size: int, config: dict):

    if gen_type.lower() == "noised_sawtooth":
        gen = NoisedSawtoothGenerator(
            torch.float32,
            seed=42,
            noise=0,
            dist_freq=UniformContinuous(1, 1),
            noise_levels=2,
            beta=0.2,
            num_context=UniformDiscrete(num_context, num_context),
            num_target=UniformDiscrete(target_size, target_size),
            **config,
        )
    elif gen_type.lower() == "noised_square_wave":
        gen = NoisedSquareWaveGenerator(
            torch.float32,
            seed=46,
            noise=0,
            noise_levels=2,
            beta=0.15,
            num_context=UniformDiscrete(num_context, num_context),
            num_target=UniformDiscrete(target_size, target_size),
            **config,
        )
    elif gen_type.lower() == "noised_gp":
        gen = NoisedGPGenerator(
            torch.float32,
            seed=45,
            noise=0,
            noise_levels=2,
            beta=0.2,
            num_context=UniformDiscrete(num_context, num_context),
            num_target=UniformDiscrete(target_size, target_size),
            **config,
        )
    else:
        raise ValueError("Selected gen_type has not been implemented.")

    batches = []
    for _ in range(num_batches):
        batches.append(gen.generate_batch())
    
    return batches


if __name__ == "__main__":


    config = {
            "num_tasks": 1,
            "batch_size": 1,
            "dist_x_context": UniformContinuous(*((X_RANGE_CONTEXT,))),
            "dist_x_target": UniformContinuous(*((X_RANGE_TARGET,))),
            "dim_y": 1,
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
            "same_xt": True,
        }

    batch = get_batches(10, 1, "noised_square_wave", 3000, config)

    dataset_plot(batch[0])