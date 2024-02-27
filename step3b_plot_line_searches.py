
import time
import torch as t
import numpy as np
import matplotlib.pyplot as plt
from models import *


if __name__ == "__main__":

    # for iters in [48800, 20000, 2000, 0]:
    for iters in [0]:

        eigvals, eigvecs = t.load(f'./models/resnet9_cifar10/eig_{iters}/eigvals_vecs.pth')
        gradient = t.load(f'./models/resnet9_cifar10/gradients/{iters}.pth')
        f_values = t.load(f'./models/resnet9_cifar10/line_searches/{iters}_top_10_largest.pth')

        n_eig = f_values.size(0)
        n_chunks = f_values.size(1)
        n_explore = f_values.size(2)

        argmin_locs = []

        n_params = eigvals.size(0)
        eigindices = list(n_params - np.exp(np.arange(0, np.log(n_params), np.log(n_params) / 30)).astype(int))
        for i in range(len(eigindices) - 1):
            if eigindices[i + 1] in eigindices[:i + 1]:
                eigindices[i + 1] = min(eigindices[:i + 1]) - 1
        eigindices.append(0)

        plotted_eigvals = eigvals[eigindices]

        for eig_index in range(len(eigindices)):

            # plt.subplots(211)
            plt.figure(figsize=(12,8))
            plt.title(f"plot for eigval = {plotted_eigvals[eig_index]}")
            plt.subplot(211)

            for i in range(n_chunks):

                plt.plot(f_values[eig_index, i].numpy(), alpha=0.3)

                min_loc = t.argmin(f_values[eig_index, i]).item()
                plt.scatter(min_loc, f_values[eig_index, i, min_loc], color='blue')

                argmin_locs.append(min_loc)

            x = f_values[eig_index].mean(0).numpy()

            plt.plot(x, alpha=1, color='red')
            plt.vlines(n_explore/2-1, ymin=x.min(), ymax=x.max())
            plt.subplot(212)

            for i in range(n_chunks):

                plt.plot((f_values[eig_index, i, 1:] - f_values[eig_index, i, :-1]).numpy(), alpha=0.3)

                argmin_locs.append(min_loc)

            x = f_values[eig_index].mean(0)

            plt.plot((x[1:] - x[:-1]).numpy(), alpha=1, color='red')

            plt.tight_layout()
            plt.savefig(f'./models/resnet9_cifar10/line_search_plots_{iters}/{eig_index}_eig_{plotted_eigvals[eig_index]:.3e}.png', dpi=300)
            plt.close()
