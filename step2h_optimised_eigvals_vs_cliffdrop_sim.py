"""
The point of this file is to plot the number of optimised eigdirections vs the cosine distance
of the gradient at that optimised theta with the cliffdrop direction

"""


import time
import torch as t
import torch.optim as optim
import torchvision
from torch.func import functional_call
from scipy.interpolate import CubicSpline, RectBivariateSpline
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator, eigsh

from torch.func import jvp, grad, vjp
from torch.autograd.functional import vhp
from models import *
import matplotlib.pyplot as plt


def get_implied_min_location_params(vec_params, gradient, eigvals, eigvecs, eigen_threshold=1e-5):

    mask = (eigvals > eigen_threshold)

    vecs = eigvecs[:, mask]
    vals = eigvals[mask]

    theta_d = vecs.T @ gradient

    min_locs = - theta_d / vals

    min_params = vec_params + vecs @ min_locs

    return min_params


def grad_model(model, data_x, data_y, loss_fn):

    # send data to gpu

    preds = model(data_x)
    loss = loss_fn(preds, data_y)
    gradients = t.autograd.grad(loss, model.parameters())
    return t.cat([x.flatten() for x in gradients]).detach().cpu()


if __name__ == "__main__":

    data_train = torchvision.datasets.CIFAR10('../datasets/', train=True, download=True)
    data_x = t.from_numpy(data_train.data).float()
    data_y = t.LongTensor(data_train.targets)

    x_mean = data_x.mean(dim=0)
    x_std = data_x.std(dim=0)

    data_x = (data_x - x_mean) / (1e-7 + x_std)
    data_x = data_x.transpose(1, 3)

    model_0 = ResNet9(3, 10, expand_factor=1)
    model_1 = ResNet9(3, 10, expand_factor=1)
    model_0.load_state_dict(t.load(f'models/resnet9_cifar10/model_{10000}.pth'))
    model_1.load_state_dict(t.load(f'models/resnet9_cifar10/model_{10500}.pth'))

    eigvals, eigvecs = t.load(f'./models/resnet9_cifar10/eig_{10000}/eigvals_vecs.pth')
    gradient = t.load(f'./models/resnet9_cifar10/gradients/{10000}.pth')
    loss_fn = nn.CrossEntropyLoss()

    # batch_indices = np.random.permutation(np.arange(data_x.shape[0]))[:1000]
    # data_x = data_x[batch_indices]
    # data_y = data_y[batch_indices]

    # ===========================================================================

    model_0_params = model_0.get_vectorized_params().clone().detach()

    y_dir = model_1.get_vectorized_params() - model_0_params

    cosine_distance = (-gradient @ y_dir) / (gradient.norm() * y_dir.norm())

    print(f"{0} optimised dir, cos dist: {cosine_distance.item():.4f}")

    # iterate over the range of eigvals:
    n_eigvals = 100

    cos_distances = [cosine_distance.item()]

    for i in range(1, n_eigvals+1):

        threshold = eigvals[-i-1]
        # find minimum after optimising i eigvals

        min_params = get_implied_min_location_params(model_0_params, gradient, eigvals, eigvecs, eigen_threshold=threshold)

        # load params in model
        for (p1, p2) in zip(model_0.parameters(), model_0.shape_vec_as_params_no_names(min_params)):
            p1.data = p2.data.clone()

        # find gradient at the minimum
        grad_at_min = -grad_model(model_0, data_x, data_y, loss_fn)

        # compute cosine distance between gradient and y_dir
        y_dir = model_1.get_vectorized_params() - min_params

        cosine_distance = (grad_at_min @ y_dir)/(grad_at_min.norm() * y_dir.norm())
        cos_distances.append(cosine_distance.item())

        print(f"{i} optimised dir, cos dist: {cosine_distance.item():.4f}")

    plt.plot(np.arange(1, n_eigvals+1), np.array(cos_distances))
    plt.title("cliffdrop-gradient cos disance vs number of optimised eigendirs")
    plt.xlabel('number of optimised eigendirections')
    plt.ylabel('cosine distance with cliff-drop direction')
