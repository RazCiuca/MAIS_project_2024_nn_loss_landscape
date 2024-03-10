

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
from HessianEigen_utils import top_k_hessian_eigen
import matplotlib
import matplotlib.pyplot as plt


if __name__ == "__main__":

    cmap = matplotlib.colormaps['Spectral']

    top_k = 1000
    iter = 10000

    eigvals, eigvecs = t.load(f"models/resnet9_cifar10/eig_{iter}/bottom_{top_k}_eigen.pth")

    n_params = eigvecs.shape[0]

    for i in range(0, top_k):

        vecs = np.sort(eigvecs[:, i] ** 2)[::-1]
        to_plot = np.cumsum(vecs)

        plt.plot(to_plot, alpha=0.1, color=cmap(i / top_k))

    vec_randn = t.randn(n_params)
    vec_randn = vec_randn/vec_randn.norm()
    plt.plot(np.cumsum(np.sort(vec_randn.numpy()**2)[::-1]), color='black')

    plt.xscale('log')
    plt.xlabel('rank')
    plt.ylabel('cumulative power')
    plt.title('cumulative sorted power of negative eigenvectors')
    plt.show()

