from typing import List
import json
import torch
import matplotlib.pyplot as plt
import numpy as np

def _plot_curve(values: List[float], fpath: str):

    cont_size = np.arange(1, len(values) + 1, 1)

    plt.figure(figsize=(10,6))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Context size", fontsize=20, labelpad=20)
    plt.ylabel("Likelihood", fontsize=20, labelpad=20)

    plt.plot(cont_size, values, linewidth=2)
    plt.ylim(0,100)
    
    plt.grid()
    plt.tight_layout()
    plt.savefig(fpath, dpi=300)
    plt.close()


if __name__ == "__main__":

    dataset = torch.load("/scratch/lb953/benchmark_datasets/benchmark_dataset_50_targets_noised_gp_1_layers.pt", map_location="cuda")

    for kind in ["pred_logpdf_diag", "pred_logpdf"]:

        # Evaluate model predictions over context sets
        json_data = {}
        for idx, batch in enumerate(dataset):
            cont_size = batch["contexts"][0][0].numel()
            if cont_size in json_data:
                json_data[cont_size] = torch.concat((json_data[cont_size], batch[kind]/batch["xt"][0][0].numel()), dim=0)
            else:
                json_data[cont_size] = batch[kind]/batch["xt"][0][0].numel()

        for cont_size in json_data.keys():
            json_data[cont_size] = np.exp(json_data[cont_size].cpu()).mean().item()

        with open(f"{kind}.json", "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        # values = [value.item() for value in json_data.values()]
        # _plot_curve(values, "images/gp_loglik.png")