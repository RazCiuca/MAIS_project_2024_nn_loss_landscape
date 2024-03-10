"""
In this file we plot a contour graph around the cliff at iteration 10000

let theta_0 be the point at iter 10000
let theta_1 bet the point at iter 10500

the y dimentions is going to be the line search in that direction

the x dimension is going to be an average over 1000 directions weighted by the hessian eigenvalues at
theta_0

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

def evaluate_model(params, model, inputs, targets, loss_fn):

    with t.no_grad():
        ave_loss = 0

        z = model.shape_vec_as_params(params)
        preds = functional_call(model, z, inputs)
        loss = loss_fn(preds, targets)
        ave_loss += loss.item()

        return ave_loss

if __name__ == "__main__":

    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

    data_train = torchvision.datasets.CIFAR10('../datasets/', train=True, download=True)
    data_x = t.from_numpy(data_train.data).float()
    data_y = t.LongTensor(data_train.targets)

    x_mean = data_x.mean(dim=0)
    x_std = data_x.std(dim=0)

    data_x = (data_x - x_mean) / (1e-7 + x_std)
    data_x = data_x.transpose(1, 3)

    model_0 = ResNet9(3, 10, expand_factor=1)
    model_0.load_state_dict(t.load(f'models/resnet9_cifar10/model_{10000}.pth'))

    eigvals, eigvecs = t.load(f'./models/resnet9_cifar10/eig_{10000}/eigvals_vecs.pth')
    gradient = t.load(f'./models/resnet9_cifar10/gradients/{10000}.pth')
    loss_fn = nn.CrossEntropyLoss()

    # ===========================================================================

    # sending things to gpu:
    data_x = data_x.to(device)
    data_y = data_y.to(device)
    model0 = model_0.to(device)
    eigvecs = eigvecs.to(device)

    average_params = model_0.get_vectorized_params()

    from HessianEigen_utils import top_k_hessian_eigen
    # eigvals_bottom, eigvecs_bottom = top_k_hessian_eigen(model_0, data_x, data_y, loss_fn, top_k=10, mode='SA', batch_size=10000)
    eigvals_bottom, eigvecs_bottom = t.load(f'models/resnet9_cifar10/eig_{10000}/bottom_eigen.pth')
    # t.save((eigvals_bottom, eigvecs_bottom),f'models/resnet9_cifar10/eig_{10000}/bottom_eigen.pth')

    eigen_grad = t.from_numpy(t.load(f"./models/resnet9_cifar10/eig_{10000}/eigen_gradient.pth"))

    y_dir = t.from_numpy(eigvecs_bottom[:, 0]).to(device)
    # y_dir = eigen_grad.float().to(device)
    # y_dir = y_dir/y_dir.norm()
    x_dir = eigvecs[:, -1]

    # ============================================================================
    # verifying eigenvalues

    # for i in range(1000):
    #
    #     dir = eigvecs[:, i]
    #     old_eigen = eigvals[i]
    #
    #     tangents = tuple(model_0.shape_vec_as_params_no_names(dir))
    #
    #     # defining the function to send to hvp
    #     def fn_to_optim(*x):
    #         z = [{name: p} for (p, name) in zip(x, model_0.param_names)]
    #
    #         preds = functional_call(model_0, z, data_x)
    #         return loss_fn(preds, data_y)
    #
    #
    #     # calling hvp with the right vector, add to result
    #     hessian_col = vhp(fn_to_optim, tuple(model_0.parameters()), tangents)[1]
    #     hessian_col = t.cat([x.flatten() for x in hessian_col])
    #     vec_sim = hessian_col/hessian_col.norm() @ dir
    #     computed_eigen = hessian_col @ dir
    #
    #     print(f"{i}-th computed eigenval = {computed_eigen.item()}, old eigen={old_eigen}, vec sim={vec_sim.item()}")

    for n in range(0, 1):

        batch_indices = np.random.permutation(np.arange(data_x.shape[0]))[:10000]
        batch_data_x = data_x[batch_indices]
        batch_data_y = data_y[batch_indices]

        x_plot = np.arange(-0.2, 0.2, 0.01)
        y_plot = np.arange(-0.3, 0.3, 1e-2)

        X,Y = np.meshgrid(x_plot, y_plot)
        Z = np.zeros(X.shape)

        start = time.time()
        counter = 1

        for i1, x in enumerate(x_plot):
            for i2,y in enumerate(y_plot):

                params = average_params + y * y_dir + x * x_dir

                ave_loss = evaluate_model(params, model_0, batch_data_x, batch_data_y, loss_fn)

                Z[i2, i1] = ave_loss

                stop = time.time()
                time_remaining = (stop - start) * ((x_plot.shape[0]*y_plot.shape[0] - counter) / counter)
                print(f"finished {x:.3f},{y:.3f},{ave_loss:.3e}, time_remaining:{time_remaining}")
                counter += 1

        z_min_log = np.log(Z.min())
        z_max_log = np.log(Z.max())
        levels = np.exp(np.arange(z_min_log, z_max_log, (z_max_log - z_min_log) / 200))

        spline = RectBivariateSpline(x_plot, y_plot, Z.T)

        fig, ax = plt.subplots()
        CS = ax.contour(X, Y, Z, levels=levels)
        # ax.clabel(CS, inline=True, fontsize=10)
        ax.set_title('Contour plot of largest eigen vs most negative eigen')
        plt.show()
        # plt.tight_layout()
        #
        # plt.savefig(
        #     f'./figures/largest_v_smallest_plots/{n}.png',
        #     dpi=300)
        # plt.close()

    # t.save((X,Y,Z), './figures/lr_cliff_plotting_data_1.pth')

