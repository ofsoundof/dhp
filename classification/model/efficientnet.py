import torch
import torch.nn as nn
from IPython import embed

def make_model(args):
    return EfficientNet(args[0])


class swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""

    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


def drop_connect(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


class SE(nn.Module):
    """Squeeze-and-Excitation block with Swish."""

    def __init__(self, in_channels, se_channels):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, se_channels, kernel_size=1, bias=True),
            swish(),
            nn.Conv2d(se_channels, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.fc(self.avg_pool(x))
        out = x * out
        return out


class InvertedResidual(nn.Module):
    """
    Mobile inverted bottleneck block w/ SE (MBConv).
    expansion + depthwise + pointwise + squeeze-excitation
    """

    def __init__(self, in_planes, out_planes, kernel_size, stride, expand_ratio=1, drop_rate=0.):
        super(InvertedResidual, self).__init__()
        self.drop_rate = drop_rate
        # Skip connection if in and out shapes are the same (MV-V2 style)
        self.has_skip = (stride == 1) and (in_planes == out_planes)

        layers = []
        # Expansion
        planes = expand_ratio * in_planes
        # embed()
        if in_planes != planes:
            layers.extend([nn.Conv2d(in_planes, planes, 1, 1, 0, bias=False),
                           nn.BatchNorm2d(planes),
                           swish()])
        # Depthwise conv
        layers.extend([nn.Conv2d(planes, planes, kernel_size, stride, (1 if kernel_size == 3 else 2), groups=planes, bias=False),
                       nn.BatchNorm2d(planes),
                       swish()])
        # SE layers
        layers.append(SE(planes, in_planes // 4))

        # Linear proj
        layers.extend([nn.Conv2d(planes, out_planes, 1, 1, 0, bias=False),
                       nn.BatchNorm2d(out_planes)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        if self.has_skip:
            if self.training and self.drop_rate > 0:
                out = drop_connect(out, self.drop_rate)
            out = out + x
        return out


cfg = {
    'num_blocks': [1, 2, 2, 3, 3, 4, 1],
    'expansion': [1, 6, 6, 6, 6, 6, 6],
    'out_planes': [16, 24, 40, 80, 112, 192, 320],
    'kernel_size': [3, 3, 5, 3, 5, 5, 3],
    'stride': [1, 2, 2, 2, 1, 2, 1],
}


class EfficientNet(nn.Module):
    def __init__(self, args):
        super(EfficientNet, self).__init__()
        self.cfg = cfg
        self.width_mult = args.width_mult
        self.data_train = args.data_train
        if args.data_train.find('CIFAR') >= 0:
            num_classes = int(args.data_train[5:])
        elif args.data_train.find('Tiny') >= 0:
            num_classes = 200
        else:
            num_classes = 1000
        stride = 2 if args.data_train == 'ImageNet' else 1
        features = [nn.Sequential(
            nn.Conv2d(3, int(32 * self.width_mult), 3, stride, 1, bias=False),
            nn.BatchNorm2d(int(32 * self.width_mult)),
            swish())]
        features.extend(self._make_layers(int(32 * self.width_mult)))
        in_planes = int(cfg['out_planes'][-1] * self.width_mult)
        out_planes = int(1280 * self.width_mult)
        features.append(nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_planes),
            swish()))
        self.features = nn.Sequential(*features)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(out_planes, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        cfg = [self.cfg[k] for k in ['expansion', 'out_planes', 'num_blocks', 'kernel_size', 'stride']]
        for i, (expansion, out_planes, num_blocks, kernel_size, stride) in enumerate(zip(*cfg)):
            if self.data_train.find('CIFAR') >= 0 and i in [1, 2]:
                stride = 1
            out_planes = int(out_planes * self.width_mult)
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(InvertedResidual(in_planes, out_planes, kernel_size, stride, expansion, drop_rate=0))
                in_planes = out_planes
        return layers

    def forward(self, x):
        out = self.avg_pool(self.features(x))
        out = out.view(out.size()[0], -1)
        out = self.classifier(out)
        return out
