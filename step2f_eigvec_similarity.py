
"""
Plotting the eigenvector similarities between different network checkpoints
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

# plotting eigvec similarity
# for the k highest eigenvalues at the next time step, plot how they relate to the eigenvalues of the past time step
if __name__ == "__main___":

    cmap = matplotlib.colormaps['Spectral']

    all_eigvals = []
    all_eigvecs = []

    # all_iters = [0, 2000, 10000, 20000, 30000, 40000, 48800]
    all_iters = [10000, 10100, 10500]

    for iters in all_iters:

        eigvals, eigvecs = t.load(f'./models/resnet9_cifar10/eig_{iters}/eigvals_vecs.pth')

        all_eigvals.append(eigvals)
        all_eigvecs.append(eigvecs)

    i = 0

    for i1,i2 in itertools.combinations(range(len(all_iters)), 2):
    # if False:

        n_eigen = 400

        # constructing which eigenvalues we plot:

        n_params = eigvals.size(0)

        eigindices = list(n_params - np.exp(np.arange(0, np.log(n_params), np.log(n_params) / n_eigen)).astype(int))

        for k in range(len(eigindices) - 1):
            if eigindices[k + 1] in eigindices[:k + 1]:
                eigindices[k + 1] = min(eigindices[:k + 1]) - 1


        eigenvals = all_eigvals[i1][eigindices]
        power = (all_eigvecs[i2][:, eigindices].T @ all_eigvecs[i1])**2
        next_step_power = t.sort(power , dim=1, descending=True).values

        for j in range(n_eigen):

            y = next_step_power[j].numpy()

            truncation = np.searchsorted(np.cumsum(y), 0.99)

            plt.plot(y[:truncation], label=f"{eigenvals[j].item():.3e}", alpha=0.6, color=cmap(j/n_eigen))

        # rand_vec = t.randn(n_params)
        # rand_vec_power = t.sort((rand_vec/rand_vec.norm() @ all_eigvecs[i1])**2, descending=True).values.numpy()
        # truncation = np.searchsorted(np.cumsum(rand_vec_power), 0.99)
        # plt.plot(rand_vec_power[:truncation], label="rand vec", alpha=1.0, color='black', linewidth=4)

        for p in [0, 0.5, 1.0]:

            rand_vec2 = (t.randn(n_params) * t.abs(all_eigvals[i1])**p) @ all_eigvecs[i1].T
            rand_vec2 /= rand_vec2.norm()
            rand_vec_power = t.sort((rand_vec2 @ all_eigvecs[i1]) ** 2, descending=True).values.numpy()
            truncation = np.searchsorted(np.cumsum(rand_vec_power), 0.99)

            # print(rand_vec_power)
            plt.plot(rand_vec_power[:truncation], label="rand vec", alpha=1.0, color='black', linewidth=3)

        # plt.legend()
        plt.title(f"transition from {all_iters[i1]} to {all_iters[i2]}")
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1e-6, 1e-1)
        plt.tight_layout()
        # plt.show()
        plt.savefig(
            f'./figures/eigvec_sim_plots/sim_{all_iters[i1]}_{all_iters[i2]}.png',
            dpi=300)
        plt.close()


# plotting eigvec power vs eigenvalue, I want
def get_slice_power(x, y, n_slices=20):

    log_x = np.log(x)
    log_x_min = log_x.min()
    log_x_max = log_x.max()
    slice_width = (log_x_max-log_x_min + 1e-3)/n_slices

    slice_midpoints = np.arange(log_x_min, log_x_max, slice_width)

    powers = []

    for mid in slice_midpoints:
        powers.append( np.sum(y[np.logical_and( log_x > mid-slice_width/2 , log_x <= mid+slice_width/2)]) )

    return np.exp(slice_midpoints), np.array(powers)

if __name__ == "__main__":

    cmap = matplotlib.colormaps['Spectral']

    all_eigvals = []
    all_eigvecs = []

    all_iters = [10000, 10100, 10500]

    for iters in all_iters:

        eigvals, eigvecs = t.load(f'./models/resnet9_cifar10/eig_{iters}/eigvals_vecs.pth')

        all_eigvals.append(eigvals)
        all_eigvecs.append(eigvecs)

    i = 0

    for i1,i2 in itertools.combinations(range(len(all_iters)), 2):
    # if False:
        print(f"doing {i1}-{i2}")
        n_eigen = 400

        # constructing which eigenvalues we plot:

        n_params = eigvals.size(0)

        eigindices = list(n_params - np.exp(np.arange(0, np.log(n_params), np.log(n_params) / n_eigen)).astype(int))

        for k in range(len(eigindices) - 1):
            if eigindices[k + 1] in eigindices[:k + 1]:
                eigindices[k + 1] = min(eigindices[:k + 1]) - 1

        eigenvals = all_eigvals[i1][eigindices]
        power = (all_eigvecs[i2][:, eigindices].T @ all_eigvecs[i1])**2
        next_step_power = power

        for j in list(range(n_eigen))[::-1]:

            y = next_step_power[j]

            z = t.sort(y, descending=True)
            y_sorted_vals, y_sorted_indices = z.values, z.indices

            truncation = np.searchsorted(np.cumsum(y_sorted_vals.numpy()), 0.99)+1

            y = y[y_sorted_indices[:truncation]].numpy()
            x = all_eigvals[i1][y_sorted_indices[:truncation]].numpy()

            # plt.scatter(x, y, label=f"{eigenvals[j].item():.3e}", alpha=0.2, color=cmap(j/n_eigen))
            x,y = get_slice_power(np.abs(x),y, n_slices=30)
            # cs = CubicSpline(x, y)
            # xs = np.arange(x.min(), x.max(), (x.max()-x.min())/1000)
            plt.plot(x,y, alpha=0.6, color=cmap(j/n_eigen))

        # plt.legend()
        plt.title(f"transition from {all_iters[i1]} to {all_iters[i2]}")
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('eigenvalue')
        plt.ylabel('power')
        plt.xlim(1e-5, 10)
        plt.ylim(1e-5, 1)
        plt.tight_layout()
        # plt.show()
        plt.savefig(
            f'./figures/eigvec_sim_power_plots/sim_{all_iters[i1]}_{all_iters[i2]}.png',
            dpi=300)
        plt.close()
