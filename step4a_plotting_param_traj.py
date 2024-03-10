
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
from HessianEigen_utils import *


if __name__ == "__main__":


    # data_x, data_y = t.load('models/resnet9_cifar10/enlarged_dataset.pth')
    cmap = matplotlib.colormaps['Spectral']

    model = ResNet9(3, 10, expand_factor=1)

    n_params = model.get_vectorized_params().size(0)

    params = []
    iters = np.arange(0, 48900, 100)

    # computing full-dataset losses
    for iter in iters:

        model.load_state_dict(t.load(f'models/resnet9_cifar10/model_{iter}.pth'))
        params.append(model.get_vectorized_params())

    eigvals, eigvecs = t.load(f"models/resnet9_cifar10/eig_{10000}/eigvals_vecs.pth")
    mask = eigvals > 1e-4
    eigvals = eigvals[mask]
    eigvecs = eigvecs[:, mask]

    params = t.stack(params, dim=0).detach()

    # params = params/params.std(dim=0)

    n_to_plot = 5
    max_iter = 300

    params = params - params[max_iter]
    params = params @ eigvecs

    for i in range(1, n_to_plot+1):
        x = params[:max_iter, i].numpy()

        plt.plot(x, alpha=0.6, color=cmap(i/n_to_plot))

    plt.show()

    # =========================================================================
    # variability ranking within the first 10000 iters
    # =========================================================================

    # total_dif = t.abs(params[99] - params[10])
    # variabilities = params[10:100].std(dim=0)
    #
    # to_plot = total_dif/variabilities
    #
    # sorted_val, indices = t.sort(to_plot, descending=True)
    #
    # plt.plot(sorted_val)
    # plt.xscale('log')
    # plt.show()


