'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import common
from IPython import embed
# from torchvision.models import resnet
def make_model(args, parent=False):
    return MobileNetV2(args[0])


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


# (expansion, out_planes, num_blocks, stride)
cfg = [(1,  16, 1, 1),
       (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
       (6,  32, 3, 2),
       (6,  64, 4, 2),
       (6,  96, 3, 1),
       (6, 160, 3, 2),
       (6, 320, 1, 1)]

cfg_imagenet = [(1,  16, 1, 1),
       (6,  24, 2, 2),
       (6,  32, 3, 2),
       (6,  64, 4, 2),
       (6,  96, 3, 1),
       (6, 160, 3, 2),
       (6, 320, 1, 1)]


class MobileNetV2(nn.Module):


    def __init__(self, args, conv3x3=common.default_conv, conv1x1=common.default_conv):
        super(MobileNetV2, self).__init__()
        self.width_mult = args.width_mult

        # num_classes = int(args.data_train[5:]) if args.data_train.find('CIFAR') >= 0 else 1000
        if args.data_train.find('CIFAR') >= 0:
            num_classes = int(args.data_train[5:])
        elif args.data_train.find('Tiny') >= 0:
            num_classes = 200
        else:
            num_classes = 1000

        if args.data_train == 'ImageNet':
            self.cfg = cfg_imagenet
        else:
            self.cfg = cfg

        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        stride = 1 if args.data_train.find('CIFAR') >= 0 else 2

        features = [ConvBNReLU(3, int(32 * self.width_mult), kernel_size=3, stride=stride)]
        features.extend(self._make_layers(in_planes=int(32 * self.width_mult)))
        features.append(ConvBNReLU(int(320 * self.width_mult), int(1280 * self.width_mult), kernel_size=1, stride=1))
        self.features = nn.Sequential(*features)
        # self.conv1 = nn.Conv2d(3, int(32 * self.width_mult), kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d((32 *self.width_mult))
        # self.layers = self._make_layers(in_planes=int(32 * self.width_mult))
        # self.conv2 = nn.Conv2d(int(320 * self.width_mult), (1280 * self.width_mult), kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn2 = nn.BatchNorm2d(int(1280 * self.width_mult))
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(int(1280 * self.width_mult), num_classes))

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            out_planes = int(out_planes * self.width_mult)
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(InvertedResidual(in_planes, out_planes, stride, expansion))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        # out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        out = out.mean([2, 3])
        out = self.classifier(out)
        return out


def test():
    net = MobileNetV2()
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
