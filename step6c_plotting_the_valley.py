"""
todo:
- plot stochasticity of the valley in a bunch of directions

"""


import time
import functools
import torch.nn as nn
import torch as t
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from models import *
from matplotlib import pyplot as plt
import matplotlib
from scipy.interpolate import CubicSpline, RectBivariateSpline
from data_augment_cifar10 import enlarge_cifar10_dataset
from HessianEigen_utils import *
from scipy.signal import savgol_filter

def get_validation(model, data_x, targets, top_k = 5):

    with t.no_grad():
        model.eval()

        top_k_accuracy = []

        preds = model(data_x)
        sorted_preds = t.argsort(preds, dim=1)

        correct_preds = sorted_preds[:, -1] == targets

        for i in range(2, top_k+2):

            accuracy = t.mean((correct_preds).float())
            top_k_accuracy.append(accuracy.item())

            correct_preds = t.logical_or(correct_preds, sorted_preds[:, -i] == targets)

        return  top_k_accuracy


# interpolation testing
if __name__ == "__main___":
    all_average_params = t.load(f'models/resnet9_cifar10/valley_finding/all_average_params_factor_0.95.pth')

    lrs = sorted(all_average_params.keys())[::-1]

    diff_norms = []

    for i in range(0, len(lrs)-1):
        norm = (all_average_params[lrs[i+1]] - all_average_params[lrs[i]] ).norm().item()
        diff_norms.append(norm)
        print(f"lr:{lrs[i]}, diff norm: {norm}")

    cumul_norms = np.cumsum(diff_norms)
    cumul_norms /= cumul_norms[-1]

    # this is going to serve as the time parameter
    cumul_norms = np.insert(cumul_norms, 0, 0)

    all_params = t.stack([all_average_params[lr] for lr in lrs], dim=1).cpu()

    # plotting individual parameters

    for i in range(0, 20):

        index = np.random.choice(all_params.size(0))

        plt.plot(cumul_norms, all_params[index].numpy())

    plt.show()


# plotting contour plot
if __name__ == "__main__":
    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

    data_train = torchvision.datasets.CIFAR10('../datasets/', train=True, download=True)
    data_test = torchvision.datasets.CIFAR10('../datasets/', train=False, download=True)

    data_x = t.from_numpy(data_train.data).float()
    data_y = t.LongTensor(data_train.targets)
    x_mean = data_x.mean(dim=0)
    x_std = data_x.std(dim=0)
    data_x = (data_x - x_mean) / (1e-7 + x_std)
    data_x = data_x.to(device).transpose(1, 3)
    data_y = data_y.to(device)

    # a random subsample of 10000
    batch_indices = np.random.permutation(np.arange(data_x.shape[0]))[:10000]
    data_x = data_x[batch_indices]
    data_y = data_y[batch_indices]

    loss_fn = nn.CrossEntropyLoss()

    # ========================================================================
    # Loading average params
    # ========================================================================

    all_average_params = t.load(f'models/resnet9_cifar10/valley_finding/all_average_params_factor_0.95.pth')

    lrs = sorted(all_average_params.keys())[::-1]

    diff_norms = []

    for i in range(0, len(lrs) - 1):
        norm = (all_average_params[lrs[i + 1]] - all_average_params[lrs[i]]).norm().item()
        diff_norms.append(norm)
        print(f"lr:{lrs[i]}, diff norm: {norm}")

    print(diff_norms)

    cumul_norms = np.cumsum(diff_norms)
    cumul_norms /= cumul_norms[-1]
    cumul_norms = np.insert(cumul_norms, 0, 0)

    all_params_raw = t.stack([all_average_params[lr] for lr in lrs], dim=0).cpu().numpy()

    all_params = savgol_filter(all_params_raw, window_length=10, polyorder=2, axis=0)
    print("finished smoothing")

    print(cumul_norms)

    interp = CubicSpline(cumul_norms, all_params, axis=0)

    def get_interpolated_point(z):
        p = t.from_numpy(interp(z)).float()

        return p

    # def get_interpolated_point(z):
    #     index = np.searchsorted(cumul_norms, z)
    #     p1 = all_params[index-1]   # all_average_params[lrs[index - 1]]
    #     p2 = all_params[index]   # all_average_params[lrs[index]]
    #
    #     alpha = (z - cumul_norms[index - 1]) / (cumul_norms[index] - cumul_norms[index - 1])
    #
    #     p = p1 * alpha + (1 - alpha) * p2
    #
    #     return p

    # ==============================================================================

    model = ResNet9(3, 10, expand_factor=1)

    starting_params = all_average_params[lrs[0]]

    # copy params into model
    for p1, p2 in zip(model.shape_vec_as_params_no_names(starting_params.detach().clone()), model.parameters()):
        p2.data = p1

    model = model.to(device)

    top_eigval, top_eigvecs = top_k_hessian_eigen(model, data_x, data_y, loss_fn, top_k=1, mode='LA')

    x_vals = np.arange(-0.4, 0.4, 0.004)
    y_vals = np.arange(0.0, 1.0, 5e-3)

    vec_x = t.from_numpy(top_eigvecs).flatten().to(device)

    # vec_x = t.randn(model.get_vectorized_params().shape, device=device)
    # vec_x = vec_x/vec_x.norm()

    def get_params(x, y):

        # first do y:
        p = get_interpolated_point(y).to(device)
        p = p + x * vec_x

        return p

    for i in range(30):

        batch_indices = np.random.permutation(np.arange(data_x.shape[0]))[:512]
        new_data_x = data_x[batch_indices]
        new_data_y = data_y[batch_indices]

        X,Y,Z = model_contour_plot(model, get_params, new_data_x, new_data_y, loss_fn, x_vals, y_vals)

        z_min_log = np.log(Z.min())
        z_max_log = np.log(Z.max())
        levels = np.exp(np.arange(z_min_log, z_max_log, (z_max_log - z_min_log) / 50))

        spline = RectBivariateSpline(x_vals, y_vals, Z)

        fig, ax = plt.subplots()
        CS = ax.contour(X, Y, Z.T, levels=levels)
        cbar = fig.colorbar(CS)
        # ax.clabel(CS, inline=True, fontsize=10)
        ax.set_title('Contour plot of narrowing valley, random batch of size 512')
        plt.savefig(f'./images/narrow_valley_plots/{i}.png')

    # plt.show()


# top spectrum down the valley
if __name__ == "__main___":
    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

    data_train = torchvision.datasets.CIFAR10('../datasets/', train=True, download=True)
    data_test = torchvision.datasets.CIFAR10('../datasets/', train=False, download=True)

    data_x = t.from_numpy(data_train.data).float()
    data_y = t.LongTensor(data_train.targets)

    x_mean = data_x.mean(dim=0)
    x_std = data_x.std(dim=0)

    data_x = (data_x - x_mean) / (1e-7 + x_std)

    test_data_x = data_test.data
    test_data_y = t.LongTensor(data_test.targets)

    test_data_x = t.from_numpy(test_data_x)
    test_data_x = test_data_x.float()
    test_data_x = (test_data_x - x_mean) / (1e-7 + x_std)

    data_x = data_x.to(device).transpose(1, 3)
    data_y = data_y.to(device)

    test_data_x = test_data_x.to(device).transpose(1, 3)
    test_data_y = test_data_y.to(device)

    loss_fn = nn.CrossEntropyLoss()

    all_average_params = t.load(f'models/resnet9_cifar10/valley_finding/all_average_params_factor_0.95.pth')

    lrs = sorted(all_average_params.keys())[::-1]

    model = ResNet9(3, 10, expand_factor=1)

    cmap = matplotlib.colormaps['Spectral']

    for i,lr in enumerate(lrs):

        params = all_average_params[lr]

        # copy params into model
        for p1, p2 in zip(model.shape_vec_as_params_no_names(params.detach().clone()), model.parameters()):
            p2.data = p1

        # evaluate it
        model = model.to(device)
        # top_k_accuracy = get_validation(model, data_x, data_y)

        top_k = 1
        local_top_eigvals, local_top_eigvecs = top_k_hessian_eigen(model, data_x, data_y, loss_fn, top_k=top_k,
                                                                   mode='SA',
                                                                   batch_size=10000)

        # plt.plot(local_top_eigvals[::-1], label=str(lr), color=cmap(i / len(lrs)))
        # print
        print(f"lr: {lr:.2e}, smallest eigen: {local_top_eigvals.sum()}")

    # plt.yscale('log')
    # plt.xscale('log')
    # plt.legend()
    #
    # plt.show()

