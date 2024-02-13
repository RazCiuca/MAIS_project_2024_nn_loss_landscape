
import time
import functools
import torch.nn as nn
import torch as t
import numpy as np
import torchvision

import torchvision.transforms as tt
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
from timm.data.auto_augment import rand_augment_transform
from PIL import Image as im
from matplotlib import pyplot as plt

def enlarge_cifar10_dataset(data_x, data_y, n_enlarge=20):
    n_data = data_x.shape[0]

    total_data_x = []
    total_data_y = []

    image_means = [int(data_x[:, :, :, 0].mean()), int(data_x[:, :, :, 1].mean()), int(data_x[:, :, :, 2].mean())]

    tfm = rand_augment_transform(
        config_str='rand-m9-n3-mstd1',
        hparams={'translate_const': 10, 'img_mean': tuple(image_means)}
    )

    for i in range(n_data):

        if i%100 == 0:
            print(f"finished augmenting im {i}/{n_data}")

        x = im.fromarray(data_x[i])
        y = data_y[i]
        for j in range(n_enlarge):
            total_data_x.append(np.array(tfm(x)))
            total_data_y.append(y)

    return np.stack(total_data_x), np.array(total_data_y)


if __name__ == "__main__":

    data_train = torchvision.datasets.CIFAR10('../datasets/', train=True, download=True)
    data_test = torchvision.datasets.CIFAR10('../datasets/', train=False, download=True)

    data_x = data_train.data
    data_y = data_train.targets

    augmented_x, augmented_y = enlarge_cifar10_dataset(data_x, data_y, n_enlarge=10)


