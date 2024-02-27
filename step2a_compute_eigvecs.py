"""
Compute The eigenvalues and eigenvectors of the full dataset at intermediate checkpoints in the trained network
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


# computing eigstuff
if __name__ == "__main__":

    all_eigvals = []

    # for iters in [10200, 10500]:
    for iters in [0, 2000, 10000, 20000, 30000, 40000, 48800]:
        folder = f'./models/resnet9_cifar10/hess_{iters}'

        hess = []

        for i in range(0, 27):
            hess.append(t.load(folder + '/hess_' + str(i)))

        hess = t.cat(hess, dim=1)

        hess = (hess + hess.T)/2

        eigvals, eigvecs = t.linalg.eigh(hess)
        print(f"finished computing eigstuff for {iters}")

        all_eigvals.append(eigvals)

        t.save((eigvals, eigvecs), f'./models/resnet9_cifar10/eig_{iters}/eigvals_vecs.pth')

        n_zeros = int((eigvals<=1e-15).sum())


