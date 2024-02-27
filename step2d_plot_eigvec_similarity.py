"""

"""

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


# plotting eigvec similarity
if __name__ == "__main___":

    all_eigvals = []
    all_eigvecs = []

    for iters in [48800, 20000, 2000, 0]:

        eigvals, eigvecs = t.load(f'./models/resnet9_cifar10/eig_{iters}/eigvals_vecs.pth')

        all_eigvals.append(eigvals)
        all_eigvecs.append(eigvecs)

    # we are asking: if we decompose current eigenvalues into the next eigenvectors and take the 10 most high powered
    # ones, how much power are we capturing? Next, plot the number of directions required to get 0.9 power
    sims_0_to_2000 = t.sort((all_eigvecs[-1].T @ all_eigvecs[-2])**2, dim=1).values[:, -10:].sum(dim=1)
    sims_2000_to_20000 = t.sort((all_eigvecs[-2].T @ all_eigvecs[-3])**2, dim=1).values[:, -10:].sum(dim=1)
    sims_20000_to_48800 = t.sort((all_eigvecs[-3].T @ all_eigvecs[-4])**2, dim=1).values[:, -10:].sum(dim=1)

    ar_0 = sims_0_to_2000[all_eigvals[-1] > 1e-4].numpy()[::-1]
    ar_1 = sims_2000_to_20000[all_eigvals[-2]>1e-4].numpy()[::-1]
    ar_2 = sims_20000_to_48800[all_eigvals[-3]>1e-4].numpy()[::-1]

    plt.scatter(np.arange(ar_0.shape[0]), ar_0, label="0_to_2000", alpha=0.3)
    plt.scatter(np.arange(ar_1.shape[0]), ar_1, label="2000_to_20000", alpha=0.3)
    plt.scatter(np.arange(ar_2.shape[0]), ar_2, label="20000_to_48800", alpha=0.3)

    plt.legend()
    plt.xscale('log')
    plt.show()
