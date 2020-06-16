import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')


class SReLU(nn.ReLU):
    """
    ReLU shifted by 0.5 as proposed in fast.ai
    https://forums.fast.ai/t/shifted-relu-0-5/41467
    (likely no visible effect)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return super().forward(x) - 0.5


class Shortcut(nn.Module):
    """
    ResNet shortcut layer

    See the code to adjust pooling properties (concatenate avg pooling by default)
    """

    def __init__(self, downsample=False):
        super().__init__()
        self.downsample = downsample

    def forward(self, x):
        if self.downsample:
            mp = F.max_pool2d(x, kernel_size=2, stride=2)
            ap = F.avg_pool2d(x, kernel_size=2, stride=2)
            x = torch.cat([ap, ap], dim=1)
        return x


class ResidualUnit(nn.Module):
    """
    Residual unit from ResNet v2
    https://arxiv.org/abs/1603.05027
    """

    def __init__(self, in_channels, out_channels, downsample=False, use_srelu=False):
        super().__init__()
        assert in_channels == out_channels if not downsample else in_channels == out_channels // 2, 'With downsampling out_channels = in_channels * 2'

        self.use_srelu = use_srelu
        activation = SReLU if use_srelu else nn.ReLU
        self.shortcut = Shortcut(downsample)
        self.stacks = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            activation(),
            nn.Conv2d(in_channels, out_channels, 3, stride=2 if downsample else 1, padding=1),

            nn.BatchNorm2d(out_channels),
            activation(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.stacks(x) + self.shortcut(x)


class ResidualBlock(nn.Module):
    """
    Block of `num_units` residual units
    """

    def __init__(self, in_channels, out_channels, num_units, downsample=False):
        super().__init__()
        self.units = nn.Sequential(
            *[ResidualUnit(in_channels if i == 0 else out_channels, out_channels, downsample=(downsample and i == 0))
              for i in range(num_units)]
        )

    def forward(self, x):
        return self.units(x)


class ResidualNetwork(nn.Module):
    """
    ResNet-18
    https://arxiv.org/abs/1603.05027
    """

    def __init__(self, num_units=3, n_classes=2, in_channels=3, n_embeddings=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1),

            ResidualBlock(in_channels=16, out_channels=16, num_units=num_units, downsample=False),
            ResidualBlock(in_channels=16, out_channels=32, num_units=num_units, downsample=True),
            ResidualBlock(in_channels=32, out_channels=64, num_units=num_units, downsample=True),

            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            # Neck
            # nn.Linear(64, n_embeddings),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm1d(n_embeddings),
        )
        self.fc = nn.Linear(n_embeddings, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.network(x)
        return self.fc(x)


def resnet18(n_classes, **kwargs):
    return ResidualNetwork(n_classes=n_classes, **kwargs)
