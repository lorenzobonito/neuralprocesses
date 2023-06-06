from typing import List

import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import rc
# rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
# rc("text", usetex=True)


def plot_train_val_curves(train_loglik: List[float], val_loglik: List[float], filename: str):

    epochs = np.arange(1, len(train_loglik) + 1, 1)

    plt.figure(figsize=(10,6))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Epoch", fontsize=20, labelpad=20)
    plt.ylabel("Log-likelihood", fontsize=20, labelpad=20)

    plt.plot(epochs, val_loglik, label="Validation", linewidth=2)
    plt.plot(epochs, train_loglik, label="Training", linewidth=2)

    plt.legend(loc="lower right", fontsize=16)
    
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"images/{filename}.png", dpi=300)
    plt.close()


if __name__ == "__main__":

    levels = [0, 1, 2]

    for level in levels:

        fpath = f"_experiments/noised_sawtooth_diff_targ/convcnp_100/{level}/log_train.txt"

        with open(fpath, "r") as f:
            lines = f.readlines()

        train_losses = []
        val_losses = []
        
        for line in lines:

            if "Loglik (T):" in line:
                train_losses.append(float(line.split()[4]))

            if "Loglik (V):" in line:
                val_losses.append(float(line.split()[4]))

        plot_train_val_curves(train_losses, val_losses, f"train_val_curves_{level}_100_epochs")