from typing import List

import numpy as np
import torch
import lab as B
import neuralprocesses.torch as nps
from neuralprocesses.data.gp import GPGenerator
from neuralprocesses.aggregate import Aggregate, AggregateInput
from neuralprocesses.dist.uniform import UniformDiscrete, UniformContinuous
import matplotlib.pyplot as plt

X_RANGE_CONTEXT = (-2, 2)
X_RANGE_TARGET = (-2, 2)

def _plot_curve(values: List[float], fpath: str):

    cont_size = np.arange(1, len(values) + 1, 1)

    plt.figure(figsize=(10,6))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Context size", fontsize=20, labelpad=20)
    plt.ylabel("Log-likelihood", fontsize=20, labelpad=20)

    plt.plot(cont_size, values, linewidth=2)
    
    plt.grid()
    plt.tight_layout()
    plt.savefig(fpath, dpi=300)
    plt.close()

def _get_batches(num_context: int, num_batches: int, target_size: int, config: dict):

    gen = GPGenerator(
        torch.float32,
        seed=42,
        noise=0,
        num_context=UniformDiscrete(num_context, num_context),
        num_target=UniformDiscrete(target_size, target_size),
        **config,
    )

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
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }

    dataset = []
    for context_size in range(26):
        dataset.extend(_get_batches(context_size, 10, 50, config))

    # torch.save(dataset, f"benchmark_dataset_original_gp.pt")

    json_data = {}
    for idx, batch in enumerate(dataset):
        cont_size = batch["contexts"][0][0].numel()
        if cont_size in json_data:
            json_data[cont_size] = torch.concat((json_data[cont_size], batch["pred_logpdf_diag"]/batch["xt"][0][0].numel()), dim=0)
        else:
            json_data[cont_size] = batch["pred_logpdf_diag"]/batch["xt"][0][0].numel()

    for cont_size in json_data.keys():
        json_data[cont_size] = json_data[cont_size].mean()
    values = [value.item() for value in json_data.values()]
    _plot_curve(values, "images/original_gp_loglik.png")