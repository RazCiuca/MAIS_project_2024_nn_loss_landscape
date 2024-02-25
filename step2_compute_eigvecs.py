

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


def grad_model(model, data_x, data_y, loss_fn):

    # send data to gpu

    preds = model(data_x)

    loss = loss_fn(preds, data_y)

    gradients = t.autograd.grad(loss, model.parameters())

    return t.cat([x.flatten() for x in gradients]).detach().cpu()

# compute model gradients
if __name__ == "__main__":

    # data_x, data_y = t.load('models/resnet9_cifar10/enlarged_dataset.pth')

    data_train = torchvision.datasets.CIFAR10('../datasets/', train=True, download=True)
    data_x = t.from_numpy(data_train.data).float()
    data_y = t.LongTensor(data_train.targets)

    x_mean = data_x.mean(dim=0)
    x_std = data_x.std(dim=0)

    data_x = (data_x - x_mean) / (1e-7 + x_std)
    data_x = data_x.transpose(1, 3)

    for iters in [10100, 10200, 10500]:

        model = ResNet9(3, 10, expand_factor=1)
        model.load_state_dict(t.load(f'models/resnet9_cifar10/model_{iters}.pth'))

        # model = model.cuda()
        loss_fn = nn.CrossEntropyLoss()

        gradient = grad_model(model, data_x, data_y, loss_fn)
        t.save(gradient, f'./models/resnet9_cifar10/gradients/{iters}.pth')



# computing eigstuff
if __name__ == "__main___":

    all_eigvals = []

    for iters in [10200, 10500]:
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



# plotting eigvals
if __name__ == "__main___":

    all_eigvals = []

    for iters in [10000, 10100, 10200, 10500]:

        eigvals, eigvecs = t.load(f'./models/resnet9_cifar10/eig_{iters}/eigvals_vecs.pth')

        n_zeros = int((eigvals<=1e-15).sum())

        eigvals = eigvals[eigvals > 1e-4]

        plt.plot(eigvals.numpy()[::-1], label=str(iters))

    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()


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
