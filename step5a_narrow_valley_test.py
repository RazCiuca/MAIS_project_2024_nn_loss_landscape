"""
pick the network at iteration 10000, find the top 2000 eigenvalues,
then iterate over them, and find the full-batch minimum in the subspace defined by the top k eigenvalues,
then compute the bottom 2000 eigenvalues at eigen k, and find the total power of the negative directions.

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

from HessianEigen_utils import *

if __name__ == "__main__":

    # ======================================================================
    # Data Loading
    # ======================================================================

    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

    data_train = torchvision.datasets.CIFAR10('../datasets/', train=True, download=True)
    data_x = t.from_numpy(data_train.data).float()
    data_y = t.LongTensor(data_train.targets)

    x_mean = data_x.mean(dim=0)
    x_std = data_x.std(dim=0)

    data_x = (data_x - x_mean) / (1e-7 + x_std)
    data_x = data_x.transpose(1, 3)

    data_x = data_x.to(device)
    data_y = data_y.to(device)

    top_k = 2000
    iter = 10000

    n_data = data_x.size(0)
    batch_size = 10000
    indices = np.random.permutation(np.arange(n_data))[:batch_size]
    data_x = data_x[indices]
    data_y = data_y[indices]


    # ======================================================================
    # Model loading
    # ======================================================================

    model = ResNet9(3, 10, expand_factor=1)
    model1 = ResNet9(3, 10, expand_factor=1)
    model.load_state_dict(t.load(f'models/resnet9_cifar10/model_{iter}.pth'))


    gradient = t.load(f'./models/resnet9_cifar10/gradients/{iter}.pth')

    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    # ======================================================================
    # Computing and saving top eigenvectors and values
    # ======================================================================
    # top_eigvals, top_eigvecs = top_k_hessian_eigen(model, data_x, data_y, loss_fn, top_k = top_k, mode='LA', batch_size=None)
    # print('finished computing top eigstuff')
    # bottom_eigvals, bottom_eigvecs = top_k_hessian_eigen(model, data_x, data_y, loss_fn, top_k=top_k, mode='SA',
    #                                                batch_size=None)

    print('finished computing bottom eigstuff')

    # save eigenvals and vectors
    # t.save((top_eigvals, top_eigvecs), "models/resnet9_cifar10/valley_testing/init_top_eigstuff.pth")
    # t.save((bottom_eigvals, bottom_eigvecs), "models/resnet9_cifar10/valley_testing/init_bottom_eigstuff.pth")

    top_eigvals, top_eigvecs = t.load("models/resnet9_cifar10/valley_testing/init_top_eigstuff.pth")
    bottom_eigvals, bottom_eigvecs = t.load("models/resnet9_cifar10/valley_testing/init_bottom_eigstuff.pth")

    print(f"total negative power: {bottom_eigvals[bottom_eigvals<0].sum():.6f}")

    # ======================================================================
    # iterate logarithmically and optimise in subspace
    # ======================================================================

    n_steps = 30

    # ns = np.ceil(np.exp(np.arange(2, np.log(top_k), np.log(top_k)/n_steps))).astype(np.int64)
    ns = [50, 100, 250, 500, 1000, 2000]

    for n in ns:

        eigvals_sub = top_eigvals[-n:]
        eigvecs_sub = top_eigvecs[:, -n:]
        new_optimal_params = sgd_in_subspace(model, loss_fn, data_x, data_y, eigvals_sub, eigvecs_sub, device, n_iter=1000)

        # ======================================================================
        # compute bottom eigenvectors and save
        # ======================================================================

        for p1, p2 in zip(model1.parameters(), new_optimal_params):
            p1.data = p2

        model1.to(device)

        bottom_eigvals, bottom_eigvecs = top_k_hessian_eigen(model1, data_x, data_y, loss_fn, top_k=top_k, mode='SA',
                                                       batch_size=None, v0=bottom_eigvecs[:, 0])

        neg_power = bottom_eigvals[bottom_eigvals < 0].sum()

        print(f"total negative power after {n}: {neg_power:.6f}")
        # save new eigenvectors
        t.save((bottom_eigvals, bottom_eigvecs), f"models/resnet9_cifar10/valley_testing/{n}_optim_bottom_eigstuff_power_{neg_power:.3f}.pth")
