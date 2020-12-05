import torch
import torch.nn as nn
import keras4torch as k4t

class _ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(_ResidualBlock, self).__init__()
        self.sequential = nn.Sequential(
            self.bn_relu_conv(out_channels, stride=stride),
            self.bn_relu_conv(out_channels, stride=1)
        )

        self.equalInOut = (in_channels == out_channels)

        if not self.equalInOut:
            self.conv_shortcut = k4t.layers.Conv2d(out_channels, kernel_size=1, stride=stride, padding=0, bias=False)

    @staticmethod
    def bn_relu_conv(channels, stride):
        return nn.Sequential(
            k4t.layers.BatchNorm2d(),
            nn.ReLU(inplace=True),
            k4t.layers.Conv2d(channels, kernel_size=3, stride=stride, padding=1, bias=False)
        )

    def forward(self, x):
        if not self.equalInOut:
            return self.conv_shortcut(x) + self.sequential(x)
        else:
            return x + self.sequential(x)

class ResidualBlock(k4t.layers.KerasLayer):
    def build(self, in_shape: torch.Size):
        return _ResidualBlock(in_shape[1], *self.args, **self.kwargs)


def stack_blocks(n, channels, stride):
    return nn.Sequential(
            *[ResidualBlock(channels, stride if i == 0 else 1) for i in range(n)]
        )


def wideresnet(depth, num_classes, widen_factor=10):
    nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
    assert((depth - 4) % 6 == 0)
    n = (depth - 4) // 6

    model = nn.Sequential(
            k4t.layers.Conv2d(nChannels[0], kernel_size=3, stride=1, padding=1, bias=False),
            stack_blocks(n, nChannels[1], stride=1),
            stack_blocks(n, nChannels[2], stride=2),
            stack_blocks(n, nChannels[3], stride=2),

            k4t.layers.BatchNorm2d(), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            k4t.layers.Linear(num_classes)
        )

    return model