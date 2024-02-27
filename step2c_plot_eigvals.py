"""
Plots the eigenvalues of already computed spectra through time
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


# plotting eigvals
if __name__ == "__main__":

    all_eigvals = []
    import matplotlib
    cmap = matplotlib.colormaps['Spectral']

    plt.subplot(121)

    for i, iters in enumerate([0, 2000, 10000, 20000, 30000, 40000]):

        eigvals, eigvecs = t.load(f'./models/resnet9_cifar10/eig_{iters}/eigvals_vecs.pth')

        n_zeros = int((eigvals<=1e-15).sum())

        eigvals = eigvals[eigvals > 1e-4]

        plt.plot(eigvals.numpy()[::-1], label=str(iters), color=cmap(i/6))

    plt.title('positive eigenvalues')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('rank')
    plt.ylabel('eigenvalue')
    plt.legend()
    plt.subplot(122)

    for i, iters in enumerate([0, 2000, 10000, 20000, 30000, 40000]):

        eigvals, eigvecs = t.load(f'./models/resnet9_cifar10/eig_{iters}/eigvals_vecs.pth')

        eigvals = eigvals[eigvals < 0]

        plt.plot(np.abs(eigvals.numpy()), label=str(iters), color=cmap(i/6))

    plt.title('negative eigenvalues')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('rank')
    plt.ylabel('eigenvalue')
    plt.legend()
    plt.show()

