"""
Computes the full dataset gradient at given checkpoints
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
