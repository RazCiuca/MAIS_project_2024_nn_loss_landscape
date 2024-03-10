"""
investigates how the loss decreases as we do partial newton steps, considering bigger and bigger subsets of the
eigenvectors

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

def get_implied_min_location_params(vec_params, gradient, eigvals, eigvecs, eigen_threshold=1e-5):

    mask = (eigvals > eigen_threshold)

    vecs = eigvecs[:, mask]
    vals = eigvals[mask]

    theta_d = vecs.T @ gradient

    min_locs = - theta_d / vals

    min_params = vec_params + vecs @ min_locs

    return min_params

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


# compute model gradients
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

    # computing full-dataset losses
    for iter in iters:

        model.load_state_dict(t.load(f'models/resnet9_cifar10/model_{iter}.pth'))
        model = model.cuda()
        loss_fn = nn.CrossEntropyLoss()

        with t.no_grad():
            preds = model(data_x)
            loss = loss_fn(preds, data_y).detach().item()

            print(f"for iter {iter}, loss is {loss: .7f}")
            losses.append(loss)

    # now computing min_locs at given points:
    eigstuff_iters = np.array([0, 500, 2000, 10000, 20000, 30000, 40000, 48800])
    min_losses = []

    for iter in eigstuff_iters:
        model.load_state_dict(t.load(f'models/resnet9_cifar10/model_{iter}.pth'))
        eigvals, eigvecs = t.load(f'./models/resnet9_cifar10/eig_{iter}/eigvals_vecs.pth')
        gradient = t.load(f'./models/resnet9_cifar10/gradients/{iter}.pth')

        model = model.cuda()
        vec_params = model.get_vectorized_params().cuda()
        eigvals = eigvals.cuda()
        eigvecs = eigvecs.cuda()
        gradient = gradient.cuda()

        # find the top n eigenvalues

        n_eigvals = eigvals.shape[0]

        thresholds = np.exp(np.arange(0, np.log(n_eigvals*0.9), np.log(n_eigvals*0.9)/20))
        # thresholds = [-11]

        min_list = []

        for i,threshold in enumerate(thresholds):

            eigen_threshold = np.abs(np.sort(eigvals.cpu().numpy())[-int(np.ceil(threshold))])

            min_params = get_implied_min_location_params(vec_params, gradient, eigvals, eigvecs, eigen_threshold=eigen_threshold)

            implied_min_loss = get_loss(min_params, model, data_x, data_y, loss_fn)

            plt.scatter(iter, implied_min_loss, color = cmap(i/len(thresholds)))

            min_list.append(implied_min_loss)

        print(f"for iter {iter}, loss is {min_list}")

    # cm = plt.colorbar(ticks=np.log(thresholds), label='eigenvalue threshold')
    # plt.clim(-0.5, 5.5)
    plt.plot(iters, np.array(losses), label='training losses')
    # plt.scatter(eigstuff_iters, np.array(min_losses), color='red', label="implied minimums")
    # plt.ylim(top=1.0)
    plt.legend()
    plt.yscale('log')
    plt.show()

