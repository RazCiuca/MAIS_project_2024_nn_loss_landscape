"""
plots power spectra of eigenvector components, figuring out how much mass they put on how few components.
"""

import itertools
import time
import torch as t
import torch.optim as optim
import torchvision
from torch.func import functional_call
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator, eigsh

from torch.func import jvp, grad, vjp
from torch.autograd.functional import vhp
from models import *
import matplotlib.pyplot as plt
import matplotlib


if __name__ == "__main__":

    cmap = matplotlib.colormaps['Spectral']

    all_eigvals = []
    all_eigvecs = []

    top_k_eigvals = 2000

    all_iters = [10000, 20000, 30000, 40000, 48800]

    n_dim = None

    for iters in all_iters:

        eigvals, eigvecs = t.load(f'./models/resnet9_cifar10/eig_{iters}/eigvals_vecs.pth')

        n_dim = eigvals.shape[0]

        all_eigvals.append(eigvals)
        all_eigvecs.append(eigvecs)

    top_power = 0.5

    for j,iter in enumerate(all_iters):

        eig_sum = 0
        eig_weighted_power = np.zeros(n_dim)

        for i in range(1, top_k_eigvals+1):

            eig_weighted_power += all_eigvals[j][-i].item() * all_eigvecs[j][:, -i].numpy()**2
            eig_sum += all_eigvals[j][-i]

            vecs = np.sort(all_eigvecs[j][:, -i].numpy()**2)[::-1]
            to_plot = np.cumsum(vecs)

            plt.plot(to_plot, alpha=0.1, color=cmap(i/top_k_eigvals))

        vecs = np.sort(np.random.randn(n_dim)**2/n_dim)[::-1]
        to_plot = np.cumsum(vecs)
        plt.plot(to_plot, color='black')

        # plt.legend()
        plt.title(f"cumul eigvec power for iter {iter}, with top {top_k_eigvals} eigvals")
        plt.ylabel('cumulative power')
        plt.xlabel('rank')
        plt.xscale('log')
        # plt.yscale('log')
        plt.ylim(1e-2, 1.0)
        plt.tight_layout()
        plt.show()
        # plt.savefig(
        #     f'./figures/eigvec_power_plots/{iter}.png',
        #     dpi=300)
        # plt.close()

        plt.title(f'eigenvalue weighted cumul power for iter {iter}')
        plt.plot(np.cumsum(np.sort(eig_weighted_power)[::-1])/eig_sum)
        plt.xscale('log')
        plt.tight_layout()
        plt.show()
        # plt.savefig(
        #     f'./figures/eigvec_power_plots/eig_weighted_{iter}.png',
        #     dpi=300)
        # plt.close()


# just random vectors
if __name__ == "__main___":

    from step3_Lanczos_optim import orthogonalize

    cmap = matplotlib.colormaps['Spectral']

    all_eigvals = []
    all_eigvecs = []

    top_k_eigvals = 2000
    n_dim = 26000

    eigvecs, norms = orthogonalize(t.randn(top_k_eigvals+1, n_dim))
    eigvecs = (eigvecs/ eigvecs.norm(dim=1).unsqueeze(1)).T
    print("finished ortho")

    eig_sum = 0
    eig_weighted_power = np.zeros(n_dim)

    for i in range(1, top_k_eigvals+1):

        vecs = np.sort(eigvecs[:, -i].numpy()**2)[::-1]
        to_plot = np.cumsum(vecs)

        plt.plot(to_plot, alpha=0.1, color=cmap(i/top_k_eigvals))

    # vecs = np.sort(np.random.randn(n_dim)**2/n_dim)[::-1]
    # to_plot = np.cumsum(vecs)
    # plt.plot(to_plot, color='black')

    # plt.legend()
    plt.title(f"cumul eigvec power for random vectors, with top {top_k_eigvals} eigvals")
    plt.xscale('log')
    # plt.yscale('log')
    plt.ylim(1e-2, 1.0)
    plt.tight_layout()
    plt.show()
    # plt.savefig(
    #     f'./figures/eigvec_power_plots/{iter}.png',
    #     dpi=300)
    # plt.close()
