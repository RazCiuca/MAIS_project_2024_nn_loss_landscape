"""
train 2 resnets on CIFAR10 initialized at same point, using AdamW

except, one has


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

import timm

def get_validation(model, loss_fn, data_x, targets, top_k = 5):

    with t.no_grad():
        model.eval()

        top_k_accuracy = []

        preds = model(data_x)
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
    # data_x, data_y = enlarge_cifar10_dataset(data_train.data, data_train.targets, n_enlarge=20)
    # data_y = t.LongTensor(data_y)
    # #
    # data_x = t.from_numpy(data_x)
    # data_x = data_x.float()
    #
    # t.save((data_x, data_y), 'models/resnet9_cifar10/enlarged_dataset.pth')
    data_x, data_y = t.load( 'models/resnet9_cifar10/enlarged_dataset.pth')

    x_mean = data_x.mean(dim=0)
    x_std = data_x.std(dim=0)

    data_x = (data_x - x_mean) / (1e-7 + x_std)

    test_data_x = data_test.data
    test_data_y = t.LongTensor(data_test.targets)

    test_data_x = t.from_numpy(test_data_x)
    test_data_x = test_data_x.float()
    test_data_x = (test_data_x - x_mean) / (1e-7 + x_std)

    lr = 1e-3
    weight_decay = 2e-3
    # grad_clipping = 1
    batch_size = 512
    n_epoch = 50
    n_data = data_x.size(0)
    n_iter = int(n_epoch * data_x.size(0) / batch_size)

    # ====================== Resnet MODEL =============================

    model_base = timm.create_model('resnet34')
    model_mod = timm.create_model('resnet34')
    model_mod_2 = timm.create_model('resnet34')

    # make both models identical
    for p1, p2 in zip(model_base.parameters(), model_mod.parameters()):
        p2.data = p1.data.clone()

    for p1, p2 in zip(model_base.parameters(), model_mod_2.parameters()):
        p2.data = p1.data.clone()

    optimizer_base = t.optim.AdamW(model_base.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_mod_2 = t.optim.AdamW(model_mod_2.parameters(), lr=lr/10, weight_decay=weight_decay)

    param_groups = []
    param_group_names = []
    for name, parameter in model_mod.named_parameters():
        # true_lr = lr/10 if (('bias' in name) or ('bn' in name) or ('fc' in name)) else lr
        true_lr = lr/10 if (('bias' in name) or ('bn' in name)) else lr*10

        param_groups.append({'params': [parameter], 'lr': true_lr})
        param_group_names.append(name)

    optimizer_mod = t.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

    loss_fn = nn.CrossEntropyLoss()

    # send things to the proper device
    model_base = model_base.to(device)
    model_mod = model_mod.to(device)
    model_mod_2 = model_mod_2.to(device)

    data_x = data_x.to(device).transpose(1, 3)
    data_y = data_y.to(device)

    test_data_x = test_data_x.to(device).transpose(1, 3)
    test_data_y = test_data_y.to(device)

    losses_base = []
    losses_mod = []
    losses_mod_2 = []

    accuracies_base = []
    accuracies_mod = []
    accuracies_mod_2 = []

    start = time.time()

    for iter in range(n_iter):

        model_base.train()
        model_mod.train()
        model_mod_2.train()

        batch_indices = np.random.permutation(np.arange(n_data))[:batch_size]

        # batch_indices_total.append(t.from_numpy(batch_indices))

        inputs = data_x[batch_indices]
        targets = data_y[batch_indices]

        # pass through model
        preds_base = model_base(inputs)
        preds_mod = model_mod(inputs)
        preds_mod_2 = model_mod_2(inputs)

        # compute loss and append to list
        loss_base = loss_fn(preds_base, targets)
        loss_mod = loss_fn(preds_mod, targets)
        loss_mod_2 = loss_fn(preds_mod_2, targets)

        losses_base.append(loss_base.item())
        losses_mod.append(loss_mod.item())
        losses_mod_2.append(loss_mod_2.item())

        # losses.append(loss.item())

        # plot_average = loss.item() if plot_average is None else plot_gamma*plot_average + (1-plot_gamma)*loss.item()

        if iter%100 == 0 and iter != 0:
            stop = time.time()

            time_remaining = (stop - start) * ((n_iter-iter)/iter)

            plot_average_base = loss_base.item()
            plot_average_mod = loss_mod.item()
            plot_average_mod_2 = loss_mod_2.item()

            # _, accuracy_base = get_validation(model_base, loss_fn, test_data_x, test_data_y, top_k = 1)
            # _, accuracy_mod = get_validation(model_mod, loss_fn, test_data_x, test_data_y, top_k=1)
            # _, accuracy_mod_2 = get_validation(model_mod_2, loss_fn, test_data_x, test_data_y, top_k=1)

            # accuracies_base.append(accuracy_base[0])
            # accuracies_mod.append(accuracy_mod[0])

            print(f"i:{iter}/{n_iter} --- time remaining:{time_remaining:.1f}s --- loss base:{plot_average_base:.7f} "
                  f"--- loss mod: {plot_average_mod:.7f} --- loss mod 2: {plot_average_mod_2:.7f} ")
                  # f" --- acc base: {accuracy_base[0]} --- acc mod: {accuracy_mod[0]}")

        # if iter%100 == 0:
        #     t.save(model.state_dict(), f'models/resnet9_cifar10/model_{iter}.pth')

        # backwards and take optim step
        optimizer_base.zero_grad()
        loss_base.backward()

        optimizer_mod.zero_grad()
        loss_mod.backward()

        optimizer_mod_2.zero_grad()
        loss_mod_2.backward()

        # nn.utils.clip_grad_value_(model.parameters(), grad_clipping)
        optimizer_base.step()
        optimizer_mod.step()
        optimizer_mod_2.step()


    plt.figure(figsize=(6, 4))

    from scipy.signal import savgol_filter

    loss_base_plot = savgol_filter(losses_base, window_length=100, polyorder=3)
    loss_mod_plot = savgol_filter(losses_mod, window_length=100, polyorder=3)
    loss_mod_2_plot = savgol_filter(losses_mod_2, window_length=100, polyorder=3)


        # plot training loss, test loss and save figure
    plt.plot(loss_base_plot, color='b', label='all params lr=1e-3')
    plt.plot(loss_mod_plot, color='orange', label='bias+BN+last at 1e-4, rest at 1e-2')
    plt.plot(loss_mod_2_plot, color='r', label='all params lr=1e-4')
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('training loss')

    plt.yscale('log')
    plt.show()



