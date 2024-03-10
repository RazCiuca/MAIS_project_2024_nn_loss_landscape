
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


def hvp_model_np(tangents_np, model, data_x, data_y, loss_fn):
    tangents = t.from_numpy(tangents_np).to('cuda').squeeze()
    tangents = tuple(model.shape_vec_as_params_no_names(tangents))

    # send data to gpu
    data_x = data_x.cuda()
    data_y = data_y.cuda()

    # defining the function to send to hvp
    def fn_to_optim(*x):
        z = [{name: p} for (p, name) in zip(x, model.param_names)]

        preds = functional_call(model, z, data_x)
        return loss_fn(preds, data_y)

    hessian_col = vhp(fn_to_optim, tuple(model.parameters()), tangents)[1]

    return t.cat([x.flatten() for x in hessian_col]).detach().cpu().numpy()

def top_k_hessian_eigen(model, data_x, data_y, loss_fn, top_k = 100, mode='LA', batch_size=None, v0=None):
    """
    computes top-k eigenvalues and eigenvectors of the hessian of model, with given data,
    possibly with finite batch size
    """
    # if finite batch size, resample this at every computation?
    if batch_size is not None:
        indices = t.randperm(data_x.size(0), device=data_x.device)[:batch_size]
        data_x = data_x[indices]
        data_y = data_y[indices]


    linop = LinearOperator((model.n_params, model.n_params),
                            matvec = lambda x: hvp_model_np(x, model, data_x, data_y, loss_fn))

    eigvals, eigvecs = eigsh(linop, k=top_k, which=mode, v0=v0)

    return eigvals, eigvecs

def get_top_eigen_gradient(model, data_x, data_y, loss_fn):

    # define an epsilon to go in every direction?, say 1.2x the parameter for that direction? this way we're roughly
    # approximate to the scale.

    # get the top eigenvalue and eigenvector at the initial point
    params = model.get_vectorized_params()

    eigvals_base, eigvecs_base = top_k_hessian_eigen(model, data_x, data_y, loss_fn, top_k = 1, mode='LA', v0=None)

    eigen_derivative = []

    start = time.time()
    for i in range(params.shape[0]):

        # build next vector:
        new_params = params.clone()
        eps = (new_params[i] * 0.3).detach()
        new_params[i] = new_params[i] + eps

        for x,y in zip(model.parameters(), model.shape_vec_as_params_no_names(new_params)):
            x.data = y

        eigvals, eigvecs = top_k_hessian_eigen(model, data_x, data_y, loss_fn, top_k = 1, mode='LA', v0=eigvecs_base)

        eigen_derivative.append((eigvals[0] - eigvals_base[0])/eps.cpu().item())
        stop = time.time()

        time_remaining = (stop - start) * ((params.shape[0] - i) / (1 if i == 0 else i))
        print(f"finished {i}/{params.shape[0]}, eigen deriv:{eigen_derivative[-1]:.6f}, eps:{eps:.6f}, time remain:{time_remaining}")

    return eigen_derivative


if __name__ == "__main__":

    # define our device, send to gpu if we have it
    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

    data_train = torchvision.datasets.CIFAR10('../datasets/', train=True, download=True)

    data_x = t.from_numpy(data_train.data).float()
    data_y = t.LongTensor(data_train.targets)

    x_mean = data_x.mean(dim=0)
    x_std = data_x.std(dim=0)

    data_x = (data_x - x_mean) / (1e-7 + x_std)

    # ====================== Resnet MODEL =============================

    model = ResNet9(3, 10, expand_factor=1)
    iter = 10000
    model.load_state_dict(t.load(f'models/resnet9_cifar10/model_{iter}.pth'))
    loss_fn = nn.CrossEntropyLoss()

    # send things to the proper device
    model = model.to(device)
    data_x = data_x.to(device).transpose(1, 3)
    data_y = data_y.to(device)

    batch_size = 1000

    indices = t.randperm(data_x.size(0), device=data_x.device)[:batch_size]
    data_x = data_x[indices]
    data_y = data_y[indices]

    eigen_gradient = get_top_eigen_gradient(model, data_x, data_y, loss_fn)

    t.save(np.array(eigen_gradient), f"./models/resnet9_cifar10/eig_{iter}/eigen_gradient.pth")
