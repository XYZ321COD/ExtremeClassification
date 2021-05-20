import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import os


def visualization(name_of_the_run, optimizer, lr, reduction_value, epoch, weights, pruning_val):

    weights = weights.cpu().detach()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.grid(which='major', color='#CCCCCC', linestyle='--')
    ax.grid(which='minor', color='#CCCCCC', linestyle=':')
    imgplot = ax.imshow(weights.T)
    fig.colorbar(imgplot, ax=ax)
    ax.set_yticks(np.arange(0.5, reduction_value, 1))
    ax.set_xticks(np.arange(0.5, 11, 1))
    if not os.path.exists('output'):
        os.makedirs('output')
    plt.savefig(f"output/{name_of_the_run}_opt{optimizer}_lr{lr}_red{reduction_value}_{pruning_val}_{epoch}.png")
