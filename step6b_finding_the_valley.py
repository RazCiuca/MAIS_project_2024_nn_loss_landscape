"""
The goal here is to ask which eigenvectors lead to generalisation

i.e. for a fixed size of loss decrease, is it better that that loss decrease come from
the high eigenvalues, the low positive eigenvalues, or the negative eigenvalues?

Here's what we want to know:

- if we simply take the average of n training checkpoints when the network is at
    equilibrium, that should be the equivalent of having optimised the network, but not
    gone down the valley.

- also compute the total training loss and evaluation error at various stopping points

- Tracing the Valley: take the equilibrium average at a given learning rate, lower the learning
  rate, take the new average, and so on. The path the averages trace is the path of the valley.
  Then we can plot valley depth against a bunch of directions to see how they vary.

- do we generalise better higher up the valley?

- plot individual parameter trajectories to see if a linear model would be enough
- then plot the interpolation down the valley vs a bunch of eigenvectors, or just random directions


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
from data_augment_cifar10 import enlarge_cifar10_dataset
from HessianEigen_utils import *


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

        return top_k_accuracy

def train_cifar10(model, device, data_train, data_test, lr, batch_size, n_iter):

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

    test_data_x = test_data_x.to(device).transpose(1, 3)
    test_data_y = test_data_y.to(device)

    # lr = 1e-2
    weight_decay = 1e-3
    # batch_size = 512
    n_data = data_x.size(0)

    # ====================== Resnet MODEL =============================

    optimizer = t.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    average_params = t.zeros(model.get_vectorized_params().shape, device=device)

    # send things to the proper device
    model = model.to(device)
    data_x = data_x.to(device).transpose(1, 3)
    data_y = data_y.to(device)

    model_ave = ResNet9(3, 10, expand_factor=1).to(device)

    local_top_eigvecs = None
    ave_net_top_eigvecs = None

    for iter in range(n_iter):

        model.train()
        batch_indices = np.random.permutation(np.arange(n_data))[:batch_size]

        inputs = data_x[batch_indices]
        targets = data_y[batch_indices]

        # pass through model
        preds = model(inputs)

        # compute loss and append to list
        loss = loss_fn(preds, targets)

        if iter % 500 == 0 and iter != 0:

            plot_average = loss_fn(model(inputs), targets)

            for p1, p2 in zip(model_ave.shape_vec_as_params_no_names(average_params.detach().clone()/iter), model_ave.parameters()):
                p2.data = p1

            # top_k_accuracy = get_validation(model, test_data_x, test_data_y)

            top_k = 1
            # local_top_eigvals, local_top_eigvecs = top_k_hessian_eigen(model, data_x, data_y, loss_fn, top_k=top_k,
            #                                                            mode='LA',
            #                                                            batch_size=2000, v0=local_top_eigvecs.flatten() if local_top_eigvecs is not None else None)
            #
            # ave_net_top_eigvals, ave_net_top_eigvecs = top_k_hessian_eigen(model_ave, data_x, data_y, loss_fn, top_k=top_k,
            #                                                            mode='LA',
            #                                                            batch_size=2000, v0=ave_net_top_eigvecs.flatten() if ave_net_top_eigvecs is not None else None)

            # print
            # print(f"i:{iter}/{n_iter} --- lr:{lr:.2e}s --- loss:{plot_average.item():.7f} --- "
            #       f"local top eigen: {local_top_eigvals.sum().item():.6f} --- ave top eigen: {ave_net_top_eigvals.sum().item():.6f}")
            print(f"i:{iter}/{n_iter} --- loss:{plot_average.item():.7f}")

        # backwards and take optim step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        average_params = average_params + model.get_vectorized_params().detach().clone()

    top_k = 1
    local_top_eigvals, local_top_eigvecs = top_k_hessian_eigen(model, data_x, data_y, loss_fn, top_k=top_k,
                                                               mode='LA',
                                                               batch_size=2000, v0=local_top_eigvecs.flatten() if local_top_eigvecs is not None else None)

    ave_net_top_eigvals, ave_net_top_eigvecs = top_k_hessian_eigen(model_ave, data_x, data_y, loss_fn, top_k=top_k,
                                                               mode='LA',
                                                               batch_size=2000, v0=ave_net_top_eigvecs.flatten() if ave_net_top_eigvecs is not None else None)

    print(f" lr:{lr:.2e}s local top eigen: {local_top_eigvals.sum().item():.6f} --- ave top eigen: {ave_net_top_eigvals.sum().item():.6f}")

    return model, (average_params/n_iter)

if __name__ == "__main__":

    # define our device, send to gpu if we have it
    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

    # data_train = torchvision.datasets.MNIST('../datasets/', train=True, download=True)
    # data_test = torchvision.datasets.MNIST('../datasets/', train=False, download=True)
    data_train = torchvision.datasets.CIFAR10('../datasets/', train=True, download=True)
    data_test = torchvision.datasets.CIFAR10('../datasets/', train=False, download=True)

    lrs = [0.95**(i) for i in range(0, 130)]

    # lrs = [1.0, 5e-1, 2.5e-1, 1e-1, 5e-2, 2.5e-2, 1e-2, 5e-3, 2.5e-3, 1e-3, 5e-4, 2.5e-4, 1e-4]
    batch_size = 512
    n_iter = 2000

    # ====================== Resnet MODEL =============================

    model = ResNet9(3, 10, expand_factor=1)

    all_average_params = {}

    # attain equilibrium before doing the averaging
    model, _ = train_cifar10(model, device, data_train, data_test, lrs[0], batch_size, 2000)

    start = time.time()
    n_lr = len(lrs)

    for iter, lr in enumerate(lrs):
        model, _ = train_cifar10(model, device, data_train, data_test, lr, batch_size, 1000)
        model, average_params = train_cifar10(model, device, data_train, data_test, lr, batch_size, n_iter)
        all_average_params[lr] = average_params
        t.save(all_average_params, f'models/resnet9_cifar10/valley_finding/all_average_params_factor_0.95.pth')

        stop = time.time()

        time_remaining = (stop - start) * ((n_lr - iter-1) / (iter+1))
        print(f'FINISHED {iter}/{len(lrs)}, time remaining: {time_remaining:.1f}')
