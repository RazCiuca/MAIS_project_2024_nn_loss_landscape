
import torch as t
import torch.nn as nn


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes, expand_factor=1):
        super().__init__()

        n1 = 4 * expand_factor
        n2 = 8 * expand_factor
        n3 = 16 * expand_factor
        n4 = 32 * expand_factor
        n5 = 64 * expand_factor

        self.conv1 = conv_block(in_channels, n1)
        self.conv2 = conv_block(n1, n2, pool=True)
        self.res1 = nn.Sequential(conv_block(n2, n2), conv_block(n2, n2))

        self.conv3 = conv_block(n2, n3, pool=True)
        self.conv4 = conv_block(n3, n4, pool=True)
        self.res2 = nn.Sequential(conv_block(n4, n4), conv_block(n4, n4))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(n5, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

if __name__ == "__main__":

    model = ResNet9(3, 10, expand_factor=2)

    n_params = t.cat([x.flatten() for x in model.parameters()]).size()
    print(n_params)
