
import itertools
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


# plotting eigvec power vs eigenvalue, I want
def get_slice_power(x, y, n_slices=20):

    log_x = np.log(x)
    log_x_min = log_x.min()
    log_x_max = log_x.max()
    slice_width = (log_x_max-log_x_min + 1e-3)/n_slices

    slice_midpoints = np.arange(log_x_min, log_x_max, slice_width)

    powers = []

    for mid in slice_midpoints:
        powers.append( np.sum(y[np.logical_and( log_x > mid-slice_width/2 , log_x <= mid+slice_width/2)]) )

    return np.exp(slice_midpoints), np.array(powers)

if __name__ == "__main__":

    model_0 = ResNet9(3, 10, expand_factor=1)
    model_1 = ResNet9(3, 10, expand_factor=1)
    model_2 = ResNet9(3, 10, expand_factor=1)
    model_0.load_state_dict(t.load(f'models/resnet9_cifar10/model_{10000}.pth'))
    model_1.load_state_dict(t.load(f'models/resnet9_cifar10/model_{10100}.pth'))

    eigvals, eigvecs = t.load(f'./models/resnet9_cifar10/eig_{10000}/eigvals_vecs.pth')
    # gradient = t.load(f'./models/resnet9_cifar10/gradients/{10000}.pth')
    loss_fn = nn.CrossEntropyLoss()

    average_params = model_0.get_vectorized_params()

    # ===========================================================================

    past_params = [model_0.get_vectorized_params()]

    for i in range(5000, 10000, 100):
        model_2.load_state_dict(t.load(f'models/resnet9_cifar10/model_{i}.pth'))
        past_params.append(model_2.get_vectorized_params())
        average_params += model_2.get_vectorized_params()

    average_params /= len(past_params)

    for i in range(len(past_params)):
        past_params[i] = past_params[i] - average_params

    z = t.stack(past_params, dim=1)

    y_dir = model_1.get_vectorized_params() - average_params

    print(f"norm of y_dir:{y_dir.norm().item():.4e}")

    # step 2: remove the component along y_dir
    y_normalized = y_dir / y_dir.norm()

    pos_mask = eigvals > 0
    neg_mask = eigvals < 0

    pos_eigvals = eigvals[pos_mask].detach().numpy()
    neg_eigvals = eigvals[neg_mask].detach().numpy()

    pos_power = ((y_normalized @ eigvecs[:, pos_mask]) ** 2).detach().numpy()
    neg_power = ((y_normalized @ eigvecs[:, neg_mask]) ** 2).detach().numpy()

    total_pos_power = pos_power.sum()
    total_neg_power = neg_power.sum()

    print(f"pos power:{total_pos_power:.3e}, neg power:{total_neg_power:.3e}")

    # plt.subplots(211)
    plt.subplot(211)
    x,y = get_slice_power(pos_eigvals, pos_power)
    plt.plot(x,y/total_pos_power)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-5, 10)
    plt.ylim(1e-4, 1)

    plt.subplot(212)
    x2, y2 = get_slice_power(np.abs(neg_eigvals), neg_power)
    plt.plot(x2,y2/total_neg_power)

    # plt.legend()
    # plt.title(f"cliffdrop power")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('eigenvalue')
    plt.ylabel('power')
    plt.xlim(1e-5, 10)
    plt.ylim(1e-4, 1)
    plt.tight_layout()
    plt.show()
    # plt.savefig(
    #     f'./figures/eigvec_sim_power_plots/sim_{all_iters[i1]}_{all_iters[i2]}.png',
    #     dpi=300)
    # plt.close()

