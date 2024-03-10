"""
Computes the negative eigenspectrum of models at specified iterations
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
from HessianEigen_utils import top_k_hessian_eigen

if __name__ == "__main__":

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

    top_k = 1000

    for iters in [10000, 0, 20000]:

        model = ResNet9(3, 10, expand_factor=1)
        model.load_state_dict(t.load(f'models/resnet9_cifar10/model_{iters}.pth'))

        model = model.to(device)
        loss_fn = nn.CrossEntropyLoss()

        eigvals, eigvecs = top_k_hessian_eigen(model, data_x, data_y, loss_fn, top_k=top_k, mode='SA', batch_size=10000)
        t.save((eigvals, eigvecs), f"models/resnet9_cifar10/eig_{iters}/bottom_{top_k}_eigen.pth")
        print(f"finished iters:{iters}")


