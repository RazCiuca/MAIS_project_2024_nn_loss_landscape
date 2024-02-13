
import torch as t
import torch.nn as nn
import numpy as np


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.GELU()]
    if pool: layers.append(nn.AvgPool2d(2))
    return nn.Sequential(*layers)


def convNext_block(in_channels, shape):
    layers = [nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3),
              nn.LayerNorm(shape),
              nn.Conv2d(in_channels, 4*in_channels, kernel_size=1, padding=0),
              nn.GELU(),
              nn.Conv2d(4*in_channels, in_channels, kernel_size=1, padding=0)]
    return nn.Sequential(*layers)


class ConvNext9_cifar10(nn.Module):
    def __init__(self, in_channels, num_classes, expand_factor=1):
        super().__init__()

        n1 = 4 * expand_factor
        n2 = 8 * expand_factor
        n3 = 16 * expand_factor
        n4 = 64 * expand_factor

        self.conv1 = nn.Conv2d(in_channels, n1, kernel_size=3, padding=1)
        self.res1 = convNext_block(n1, shape=(n1, 32, 32))
        self.conv2 = nn.Sequential(nn.Conv2d(n1, n2, kernel_size=3, padding=1), nn.AvgPool2d(2))
        self.res2 = convNext_block(n2, shape=(n2, 16, 16))
        self.conv3 = nn.Sequential(nn.Conv2d(n2, n3, kernel_size=3, padding=1), nn.AvgPool2d(2))
        self.res3 = convNext_block(n3,  shape=(n3, 8, 8))

        self.classifier = nn.Sequential(nn.AvgPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(n4, num_classes))

        self.compute_param_shapes()

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.res1(out) + out
        out = self.conv2(out)
        out = self.res2(out) + out
        out = self.conv3(out)
        out = self.res3(out) + out
        out = self.classifier(out)
        return out

    def compute_param_shapes(self):
        self.param_shapes = [x[1].shape for x in self.named_parameters()]
        self.n_params = sum([np.prod(x) for x in self.param_shapes])
        self.param_names = [x[0] for x in self.named_parameters()]

    def get_vectorized_params(self):
        return t.cat([x[1].flatten() for x in self.named_parameters()])

    def shape_vec_as_params(self, vec):
        params = []
        index = 0

        for x_shape, name in zip(self.param_shapes, self.param_names):
            n = np.prod(x_shape)
            params.append({name: vec[index:n + index].reshape(x_shape)})
            index += n

        return params

    def shape_vec_as_params_no_names(self, vec):
        params = []
        index = 0

        for x_shape, name in zip(self.param_shapes, self.param_names):
            n = np.prod(x_shape)
            params.append(vec[index:n + index].reshape(x_shape))
            index += n

        return params





class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes, expand_factor=1):
        super().__init__()

        n1 = 4 * expand_factor
        n2 = 8 * expand_factor
        n3 = 16 * expand_factor
        n4 = 32 * expand_factor

        self.conv1 = conv_block(in_channels, n1)
        self.conv2 = conv_block(n1, n2, pool=True)
        self.res1 = nn.Sequential(conv_block(n2, n2), conv_block(n2, n2))

        self.conv3 = conv_block(n2, n3, pool=True)
        self.conv4 = conv_block(n3, n4, pool=True)
        self.res2 = nn.Sequential(conv_block(n4, n4), conv_block(n4, n4))

        self.classifier = nn.Sequential(nn.AvgPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(n4, num_classes))

        self.compute_param_shapes()

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

    def compute_param_shapes(self):
        self.param_shapes = [x[1].shape for x in self.named_parameters()]
        self.n_params = sum([np.prod(x) for x in self.param_shapes])
        self.param_names = [x[0] for x in self.named_parameters()]

    def get_vectorized_params(self):
        return t.cat([x[1].flatten() for x in self.named_parameters()])

    def shape_vec_as_params(self, vec):
        params = []
        index = 0

        for x_shape, name in zip(self.param_shapes, self.param_names):
            n = np.prod(x_shape)
            params.append({name: vec[index:n+index].reshape(x_shape)})
            index += n

        return params

    def shape_vec_as_params_no_names(self, vec):
        params = []
        index = 0

        for x_shape, name in zip(self.param_shapes, self.param_names):
            n = np.prod(x_shape)
            params.append(vec[index:n + index].reshape(x_shape))
            index += n

        return params

if __name__ == "__main__":

    model = ConvNext9_cifar10(3, 10, expand_factor=2)

    n_params = t.cat([x.flatten() for x in model.parameters()]).size()
    print(n_params)

    x = t.randn(100, 3, 32, 32)

    y = model(x)

