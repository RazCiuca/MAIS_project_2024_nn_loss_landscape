"""
This file trains a small Resnet on cifar10 and saves all intermediate models in the models folder

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


def get_validation(model, loss_fn, data_x, targets, top_k = 5):

    with t.no_grad():
        model.eval()

        top_k_accuracy = []

        chunk_size = 512
        n_data = data_x.size(0)

        preds = t.cat([model(data_x[i*chunk_size:(i+1)*chunk_size]) for i in range(int(n_data/chunk_size)+1) ], dim=0)
        validation_loss = loss_fn(preds, targets)

        sorted_preds = t.argsort(preds, dim=1)

        correct_preds = sorted_preds[:, -1] == targets

        for i in range(2, top_k+2):

            accuracy = t.mean((correct_preds).float())
            top_k_accuracy.append(accuracy.item())

            correct_preds = t.logical_or(correct_preds, sorted_preds[:, -i] == targets)

        return validation_loss.item(), top_k_accuracy

if __name__ == "__main__":

    # define our device, send to gpu if we have it
    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

    # data_train = torchvision.datasets.MNIST('../datasets/', train=True, download=True)
    # data_test = torchvision.datasets.MNIST('../datasets/', train=False, download=True)
    data_train = torchvision.datasets.CIFAR10('../datasets/', train=True, download=True)
    data_test = torchvision.datasets.CIFAR10('../datasets/', train=False, download=True)

    # expand dataset with augmented images
    # data_x, data_y = enlarge_cifar10_dataset(data_train.data, data_train.targets, n_enlarge=5)
    # data_y = t.LongTensor(data_y)
    #
    # data_x = t.from_numpy(data_x)
    # data_x = data_x.float()

    # t.save((data_x, data_y), 'models/resnet9_cifar10/enlarged_dataset.pth')
    data_x, data_y = t.load( 'models/resnet9_cifar10/enlarged_dataset.pth')

    # data_x = t.from_numpy(data_train.data).float()
    # data_y = t.LongTensor(data_train.targets)

    x_mean = data_x.mean(dim=0)
    x_std = data_x.std(dim=0)

    data_x = (data_x - x_mean) / (1e-7 + x_std)

    test_data_x = data_test.data
    test_data_y = t.LongTensor(data_test.targets)

    test_data_x = t.from_numpy(test_data_x)
    test_data_x = test_data_x.float()
    test_data_x = (test_data_x - x_mean) / (1e-7 + x_std)

    lr = 0.1
    weight_decay = 1e-3
    # grad_clipping = 0.1
    batch_size = 512
    n_epoch = 500
    n_data = data_x.size(0)
    n_iter = int(n_epoch * data_x.size(0) / batch_size)

    # ====================== Resnet MODEL =============================

    model = ResNet9(3, 10, expand_factor=8)
    print(f"number of parameters in model: {model.get_vectorized_params().shape}")
    optimizer = t.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # send things to the proper device
    model = model.to(device)
    data_x = data_x.to(device).transpose(1, 3)
    data_y = data_y.to(device)

    test_data_x = test_data_x.to(device).transpose(1, 3)
    test_data_y = test_data_y.to(device)

    # plotting stuff
    plot_intervals = 100

    plt.figure(figsize=(6, 4))
    plot_average = None
    plot_gamma = 0.995

    losses = []
    batch_indices_total = []

    plot_iters = []
    plot_losses = []
    validation_loss = []
    last_accuracy = [0]

    start = time.time()

    all_indices = np.random.permutation(np.arange(n_data))
    all_indices = np.concatenate([all_indices, all_indices[:batch_size].copy()])

    for iter in range(n_iter):

        model.train()

        # sample batch, send to gpu
        n_diff_between_batch = batch_size
        i1 = (iter*n_diff_between_batch) % n_data
        i2 = i1 + batch_size
        batch_indices = all_indices[i1:i2]

        # batch_indices_total.append(t.from_numpy(batch_indices))

        inputs = data_x[batch_indices]
        targets = data_y[batch_indices]

        # pass through model
        preds = model(inputs)

        # compute loss and append to list
        loss = loss_fn(preds, targets)

        # losses.append(loss.item())

        # plot_average = loss.item() if plot_average is None else plot_gamma*plot_average + (1-plot_gamma)*loss.item()

        if iter%100 == 0 and iter != 0:
            stop = time.time()

            time_remaining = (stop - start) * ((n_iter-iter)/iter)

            plot_average = loss_fn(model(inputs), targets)

            top_k = 1
            if iter % 5000 == 0:
                local_top_eigvals, local_top_eigvecs = top_k_hessian_eigen(model, data_x, data_y, loss_fn,
                                                                                 top_k=top_k,
                                                                                 mode='LA',
                                                                                 batch_size=10000,
                                                                                 chunk_size=512)
                local_bottom_eigvals, local_bottom_eigvecs = top_k_hessian_eigen(model, data_x, data_y, loss_fn, top_k=top_k,
                                                                           mode='SA',
                                                                           batch_size=10000,
                                                                           chunk_size=512)
            else:
                local_top_eigvals = [t.zeros(1)]
                local_bottom_eigvals = [t.zeros(1)]

            print(f"i:{iter}/{n_iter} --- time remaining:{time_remaining:.1f}s --- loss:{plot_average.item():.7f} "
                  f"--- top-1:{last_accuracy[0]:.4f} top eigen: {local_top_eigvals[0].item():.6f}, expected: {2/lr:.3f}"
                  f" bottom eigen: {local_bottom_eigvals[0].item():.6f}")

        # if iter%100 == 0:
        #     t.save(model.state_dict(), f'models/resnet9_cifar10/model_{iter}.pth')


        # changing learning rate once in a while
        if iter == 10000:
            for g in optimizer.param_groups:
                g['lr'] /= 10

            lr /= 10
            # optimizer = t.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=weight_decay)

        if iter == 20000:
            for g in optimizer.param_groups:
                g['lr'] /= 10

            lr /= 10
            # optimizer = t.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=weight_decay)

        if iter == 30000:
            for g in optimizer.param_groups:
                g['lr'] /= 10
            lr /= 10
            # optimizer = t.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=weight_decay)

        if iter == 40000:
            for g in optimizer.param_groups:
                g['lr'] /= 10
            lr /= 10
            # optimizer = t.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=weight_decay)

        # backwards and take optim step
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_value_(model.parameters(), grad_clipping)
        optimizer.step()

        if iter % plot_intervals == 0:
            plot_iters.append(iter)
            if plot_average is not None:
                plot_losses.append(plot_average.item())
            v_loss, accuracy = get_validation(model, loss_fn, test_data_x, test_data_y)
            last_accuracy = accuracy
            validation_loss.append(v_loss)

        # compute test set error once in a while, append to lists
    # t.save(t.tensor(losses), f'models/resnet9_cifar10/model_losses.pth')
    # t.save(t.stack(batch_indices_total), f'models/resnet9_cifar10/all_indices.pth')

    # plot training loss, test loss and save figure
    plt.plot(np.array(plot_iters[:-1]), np.array(plot_losses), color='b')
    plt.yscale('log')
    plt.savefig('./figures/small_resnet_cifar10_sgd_augmented_no_clip.png')

