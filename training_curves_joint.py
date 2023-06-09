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

    levels = (0, 1, 2)
    train_losses = {}
    val_losses = {}

    for level in levels:
        train_losses[level] = []
        val_losses[level] = []

    fpath = "_experiments/noised_sawtooth_diff_targ/x1_y3/convcnp/unet/sl_loglik/500/log_train.txt"

    with open(fpath, "r") as f:
        lines = f.readlines()

    for level in levels:
        for line in lines:
            # print(line.split())

            if f"Loglik (T, {level}):" in line:
                train_losses[level].append(float(line.split()[5]) if level == 0 else float(line.split()[4]))

            if f"Loglik (V, {level}):" in line:
                val_losses[level].append(float(line.split()[5]) if level == 0 else float(line.split()[4]))

    for level in levels:
        plot_train_val_curves(train_losses[level], val_losses[level], f"train_val_curves_new_joint_{level}_500_epochs")