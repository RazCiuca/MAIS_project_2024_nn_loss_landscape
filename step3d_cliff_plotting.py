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

        for i in range(params.size(1)):

            z = model.shape_vec_as_params(params[:, i])
            preds = functional_call(model, z, inputs)
            loss = loss_fn(preds, targets)
            ave_loss += loss.item()

        return ave_loss/params.size(1)

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
    model_2 = ResNet9(3, 10, expand_factor=1)
    model_0.load_state_dict(t.load(f'models/resnet9_cifar10/model_{10000}.pth'))
    model_1.load_state_dict(t.load(f'models/resnet9_cifar10/model_{10100}.pth'))

    eigvals, eigvecs = t.load(f'./models/resnet9_cifar10/eig_{10000}/eigvals_vecs.pth')
    gradient = t.load(f'./models/resnet9_cifar10/gradients/{10000}.pth')
    loss_fn = nn.CrossEntropyLoss()

    average_params = model_0.get_vectorized_params()

    batch_indices = np.random.permutation(np.arange(data_x.shape[0]))[:1000]
    data_x = data_x[batch_indices]
    data_y = data_y[batch_indices]

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
    print(f"Mean-Squared norm of oscillations:{ (t.mean(z.norm(dim=0)**2)**0.5).item() :.4e}")

    # step 2: remove the component along y_dir
    y_normalized = y_dir / y_dir.norm()
    z -= ((y_normalized @ z).unsqueeze(1) @ y_normalized.unsqueeze(0)).T
    # here z has shape [n_params, n_vecs]
    print(z.shape)

    x_plot = np.arange(-1.5, 1.5, 0.1)
    y_plot = np.arange(-1.5, 1.5, 0.1)

    X,Y = np.meshgrid(x_plot, y_plot)
    Z = np.zeros(X.shape)

    start = time.time()
    counter = 1

    for i1, x in enumerate(x_plot):
        for i2,y in enumerate(y_plot):

            params = average_params + y * y_dir

            all_params = params.unsqueeze(1) + x*z

            ave_loss = evaluate_model(all_params, model_0, data_x, data_y, loss_fn)

            Z[i2, i1] = ave_loss

            stop = time.time()
            time_remaining = (stop - start) * ((x_plot.shape[0]*y_plot.shape[0] - counter) / counter)
            print(f"finished {x:.3f},{y:.3f},{ave_loss:.3e}, time_remaining:{time_remaining}")
            counter += 1

    z_min_log = np.log(Z.min())
    z_max_log = np.log(Z.max())
    levels = np.exp(np.arange(z_min_log, z_max_log, (z_max_log - z_min_log) / 20))

    spline = RectBivariateSpline(x_plot, y_plot, Z.T)

    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z, levels=levels)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_title('Contour plot of cliff drop at lr decrease')
    plt.show()

    t.save((X,Y,Z), './figures/lr_cliff_plotting_data_1.pth')

