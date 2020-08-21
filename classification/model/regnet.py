import torch
import torch.nn as nn
import torch.nn.functional as F


def make_model(args):
    return RegNet(args[0])


cfg_x_200mf = {
        'depths': [1, 1, 4, 7],
        'widths': [24, 56, 152, 368],
        'strides': [2, 2, 2, 2],
        'group_width': 8,
        'bottleneck_ratio': 1,
        'se_reduction': 1,
    }
cfg_x_400mf = {
        'depths': [1, 2, 7, 12],
        'widths': [32, 64, 160, 384],
        'strides': [2, 2, 2, 2],
        'group_width': 16,
        'bottleneck_ratio': 1,
        'se_reduction': 1,
    }
cfg_x_600mf = {
        'depths': [1, 3, 5, 7],
        'widths': [48, 96, 240, 528],
        'strides': [2, 2, 2, 2],
        'group_width': 24,
        'bottleneck_ratio': 1,
        'se_reduction': 1,
    }
cfg_x_800mf = {
        'depths': [1, 3, 7, 5],
        'widths': [64, 128, 288, 672],
        'strides': [2, 2, 2, 2],
        'group_width': 16,
        'bottleneck_ratio': 1,
        'se_reduction': 1,
    }
cfg_x_4gf = {
        'depths': [2, 5, 14, 2],
        'widths': [80, 240, 560, 1360],
        'strides': [2, 2, 2, 2],
        'group_width': 40,
        'bottleneck_ratio': 1,
        'se_reduction': 1,
    }
cfg_x_8gf = {
        'depths': [2, 5, 15, 1],
        'widths': [80, 240, 720, 1920],
        'strides': [2, 2, 2, 2],
        'group_width': 120,
        'bottleneck_ratio': 1,
        'se_reduction': 1,
    }

cfg_y_200mf = {
        'depths': [1, 1, 4, 7],
        'widths': [24, 56, 152, 368],
        'strides': [2, 2, 2, 2],
        'group_width': 8,
        'bottleneck_ratio': 1,
        'se_reduction': 4,
    }
cfg_y_400mf = {
        'depths': [1, 3, 6, 6],
        'widths': [48, 104, 208, 440],
        'strides': [2, 2, 2, 2],
        'group_width': 8,
        'bottleneck_ratio': 1,
        'se_reduction': 4,
    }
cfg_y_600mf = {
        'depths': [1, 3, 7, 4],
        'widths': [48, 112, 256, 608],
        'strides': [2, 2, 2, 2],
        'group_width': 16,
        'bottleneck_ratio': 1,
        'se_reduction': 4,
    }
cfg_y_800mf = {
        'depths': [1, 3, 8, 2],
        'widths': [64, 128, 320, 768],
        'strides': [2, 2, 2, 2],
        'group_width': 16,
        'bottleneck_ratio': 1,
        'se_reduction': 4,
    }


class SE(nn.Module):
    """Squeeze-and-Excitation block with Swish."""

    def __init__(self, in_channel, se_channels):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channel, se_channels, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(se_channels, in_channel, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.fc(self.avg_pool(x))
        out = x * out
        return out


class ResBasicBlock(nn.Module):
    """Residual basic block: x + F(x), F = basic transform"""
    def __init__(self, w_in, w_out, stride, group_width, bottleneck_ratio, reduction):
        super(ResBasicBlock, self).__init__()
        # 1x1
        layers = []
        w_b = int(round(w_out * bottleneck_ratio))
        layers.extend([nn.Conv2d(w_in, w_b, 1, 1, 0, bias=False),
                       nn.BatchNorm2d(w_b),
                       nn.ReLU(inplace=True)])
        # 3x3
        num_groups = w_b // group_width if w_b % group_width == 0 else 1
        layers.extend([nn.Conv2d(w_b, w_b, 3, stride, 1, groups=num_groups, bias=False),
                       nn.BatchNorm2d(w_b),
                       nn.ReLU(inplace=True)])
        # se
        with_se = reduction > 1
        if with_se:
            layers.append(SE(w_b, w_in // reduction))
        # 1x1
        layers.extend([nn.Conv2d(w_b, w_out, 1, 1, 0, bias=False),
                       nn.BatchNorm2d(w_out)])
        self.conv = nn.Sequential(*layers)

        self.skip_proj = None
        if stride != 1 or w_in != w_out:
            self.skip_proj = nn.Sequential(
                nn.Conv2d(w_in, w_out, 1, stride, 0, bias=False),
                nn.BatchNorm2d(w_out)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        if self.skip_proj is None:
            out += x
        else:
            out += self.skip_proj(x)
        out = self.relu(out)
        return out


class RegNet(nn.Module):
    def __init__(self, args):
        super(RegNet, self).__init__()
        self.cfg = eval('cfg_' + args.regime)
        self.width_mult = args.width_mult
        self.data_train = args.data_train
        if args.data_train.find('CIFAR') >= 0:
            num_classes = int(args.data_train[5:])
        elif args.data_train.find('Tiny') >= 0:
            num_classes = 200
        else:
            num_classes = 1000

        features = []
        stride = 2 if args.data_train == 'ImageNet' else 1
        self.in_planes = int(32 * self.width_mult)
        features.extend([nn.Conv2d(3, self.in_planes, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(self.in_planes),
                         nn.ReLU(inplace=True)])
        features.extend(self._make_layer(0))
        features.extend(self._make_layer(1))
        features.extend(self._make_layer(2))
        features.extend(self._make_layer(3))
        self.features = nn.Sequential(*features)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(int(self.width_mult * self.cfg['widths'][-1]), num_classes)

    def _make_layer(self, idx):
        depth = self.cfg['depths'][idx]
        width = int(self.width_mult * self.cfg['widths'][idx])
        stride = self.cfg['strides'][idx]
        group_width = self.cfg['group_width']
        bottleneck_ratio = self.cfg['bottleneck_ratio']
        reduction = self.cfg['se_reduction']
        layers = []
        for i in range(depth):
            if self.data_train.find('CIFAR') >= 0 and idx in [0, 1]:
                s = 1
            else:
                s = stride if i == 0 else 1

            layers.append(ResBasicBlock(self.in_planes, width, s, group_width, bottleneck_ratio, reduction))
            self.in_planes = width
        return layers

    def forward(self, x):
        out = self.avg_pool(self.features(x))
        out = out.view(out.size()[0], -1)
        out = self.classifier(out)
        return out