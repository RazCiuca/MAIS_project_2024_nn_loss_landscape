"""
Load up all the model checkpoints and evaluate the loss on the whole dataset, and find the biggest eigenvalue, plotting
this

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
import matplotlib
from HessianEigen_utils import *

def get_loss(params, model, data_x, data_y, loss_fn):

    with t.no_grad():
        z = model.shape_vec_as_params(params)
        preds = functional_call(model, z, data_x)
        loss = loss_fn(preds, data_y)

    return loss.item()

def line_search_in_grad_direction(params, model, data_x, data_y, loss_fn):

    params = params.clone().detach()
    params.requires_grad = True

    z = model.shape_vec_as_params(params)
    preds = functional_call(model, z, data_x)
    loss = loss_fn(preds, data_y)

    loss.backward()


if __name__ == "__main__":

    # data_x, data_y = t.load('models/resnet9_cifar10/enlarged_dataset.pth')
    cmap = matplotlib.colormaps['Spectral']

    data_train = torchvision.datasets.CIFAR10('../datasets/', train=True, download=True)
    data_x = t.from_numpy(data_train.data).float()
    data_y = t.LongTensor(data_train.targets)

    x_mean = data_x.mean(dim=0)
    x_std = data_x.std(dim=0)

    data_x = (data_x - x_mean) / (1e-7 + x_std)
    data_x = data_x.transpose(1, 3)

    data_x = data_x.cuda()
    data_y = data_y.cuda()

    model = ResNet9(3, 10, expand_factor=1)
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    iters = np.arange(0, 48900, 100)
    top_eigvals = []

    # computing full-dataset losses
    for iter in iters:

        model.load_state_dict(t.load(f'models/resnet9_cifar10/model_{iter}.pth'))
        model = model.cuda()
        loss_fn = nn.CrossEntropyLoss()

        with t.no_grad():
            preds = model(data_x)
            loss = loss_fn(preds, data_y).detach().item()
            losses.append(loss)

        # computing top eigenvalue:
        eigvals, eigvecs = top_k_hessian_eigen(model, data_x, data_y, loss_fn, top_k=1, batch_size=20000, mode='LA')

        print(f"for iter {iter}, loss is {loss: .7f}, eigval is {eigvals.item():.3e}")
        top_eigvals.append(eigvals.item())

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_title("Full batch loss and max eigval through training")
    ax1.set_xlabel('iter')
    ax1.set_ylabel('loss', color=color)
    ax1.plot(iters, np.array(losses), color=color)
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('eigval', color=color)
    ax2.set_yscale('log')
    ax2.plot(iters, np.array(top_eigvals), color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    fig.show()

    t.save((iters, losses), './figures/iters_losses_through_training.pth')
    t.save((iters, top_eigvals), './figures/iters_top_eigvals_through_training.pth')
