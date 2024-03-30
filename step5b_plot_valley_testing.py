
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

if __name__ == "__main___":

    cmap = matplotlib.colormaps['Spectral']

    files = ["0_optim_bottom_eigstuff.pth",
             "12_optim_bottom_eigstuff_power_-30.355.pth",
             "25_optim_bottom_eigstuff_power_-25.323.pth",
             "50_optim_bottom_eigstuff_power_-20.973.pth",
             "100_optim_bottom_eigstuff_power_-17.963.pth",
             "250_optim_bottom_eigstuff_power_-14.912.pth",
             "500_optim_bottom_eigstuff_power_-13.635.pth",
             "1000_optim_bottom_eigstuff_power_-13.640.pth"]

    for i in range(len(files)):
        file = files[i]
        bottom_eigvals, bottom_eigvecs = t.load('models/resnet9_cifar10/valley_testing/' + file)

        neg_eigvals = bottom_eigvals[bottom_eigvals < 0]

        plt.plot(-neg_eigvals, label=file.split('_')[0], color=cmap(i/len(files)))

    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('eigenvalue rank')
    plt.ylabel('eigenvalue')
    plt.title('negative eigenspectra for multiple values of optimised high eigenvalues')
    plt.show()


if __name__ == "__main__":

    cmap = matplotlib.colormaps['Spectral']

    files = ["init_top_eigstuff.pth",
             "12_optim_top_eigstuff_power_67.493.pth",
             "25_optim_top_eigstuff_power_66.706.pth",
             "50_optim_top_eigstuff_power_64.531.pth",
             "100_optim_top_eigstuff_power_62.343.pth",
             "250_optim_top_eigstuff_power_59.162.pth",
             "500_optim_top_eigstuff_power_59.336.pth",
             "1000_optim_top_eigstuff_power_72.095.pth",
             "2000_optim_top_eigstuff_power_171.508.pth"]

    for i in range(len(files)):
        file = files[i]
        top_eigvals, top_eigvecs = t.load('models/resnet9_cifar10/valley_testing/' + file)

        top_eigvals = top_eigvals[::-1][:100]

        plt.plot(top_eigvals, label=file.split('_')[0], color=cmap(i/len(files)))

    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('eigenvalue rank')
    plt.ylabel('eigenvalue')
    plt.title('positive eigenspectra for multiple values of optimised high eigenvalues')
    plt.show()