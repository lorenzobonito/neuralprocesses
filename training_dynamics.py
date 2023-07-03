from typing import List

import numpy as np
import matplotlib.pyplot as plt
from wbml.experiment import WorkingDirectory
# from matplotlib import rc
# rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
# rc("text", usetex=True)

__all__ = ["joint_training_dynamics", "split_training_dynamics"]

def _plot_train_val_curves(train_loglik: List[float], val_loglik: List[float], fpath: str):

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
    plt.savefig(fpath, dpi=300)
    plt.close()


def joint_training_dynamics(wd: WorkingDirectory):

    with open(wd.file("log_train.txt"), "r") as f:
        lines = f.readlines()

    for line in lines:
        if "dim_y:" in line:
            num_layers = int(line.split()[2])
            break
    
    train_losses = {}
    val_losses = {}
    for layer in range(num_layers):
        train_losses[layer] = []
        val_losses[layer] = []

    for line in lines:
        if "Loglik (T," in line:
            if line.split()[0] != "|":
                train_losses[int(line.split()[4][0])].append(float(line.split()[5]))
            else:
                train_losses[int(line.split()[3][0])].append(float(line.split()[4]))

        if "Loglik (V," in line:
            if line.split()[0] != "|":
                val_losses[int(line.split()[4][0])].append(float(line.split()[5]))
            else:
                val_losses[int(line.split()[3][0])].append(float(line.split()[4]))
    
    for layer in range(len(train_losses.keys())):
        _plot_train_val_curves(train_losses[layer], val_losses[layer], wd.file(f"images/dynamics/train_val_curves_{layer}.png"))


def split_training_dynamics(wd: WorkingDirectory):

    with open(wd.file("log_train.txt"), "r") as f:
        lines = f.readlines()
    
    train_losses = []
    val_losses = []

    for line in lines:
        if "Loglik (T)" in line:
                train_losses.append(float(line.split()[4]))
        if "Loglik (V)" in line:
            if line.split()[0] != "|":
                val_losses.append(float(line.split()[4]))
            else:
                val_losses.append(float(line.split()[3]))
    
    _plot_train_val_curves(train_losses, val_losses, wd.file(f"images/dynamics/train_val_curves.png"))
