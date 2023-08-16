from typing import List
import torch
import matplotlib.pyplot as plt
import numpy as np


def _plot_curve(x: List[int], y: List[int], fpath: str):

    plt.figure(figsize=(10,6))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("# AR Samples", fontsize=20, labelpad=20)
    plt.ylabel("Time (s)", fontsize=20, labelpad=20)

    plt.plot(x, y, linewidth=2)
    
    plt.grid()
    plt.tight_layout()
    plt.savefig(fpath, dpi=300)
    plt.close()


if __name__ == "__main__":

    _plot_curve([100, 200, 300, 400, 500, 1000, 1500], [8, 23, 50, 83, 125, 499, 1123], "runtime.png")