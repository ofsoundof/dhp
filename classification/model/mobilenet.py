'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import common
from model.mobilenetv2 import ConvBNReLU
from IPython import embed


def make_model(args, parent=False):
    return MobileNet(args[0])

class DepthwiseSeparableConv(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        conv = [ConvBNReLU(in_planes, in_planes, kernel_size=3, stride=stride, groups=in_planes),
                ConvBNReLU(in_planes, out_planes, kernel_size=1, stride=1)]
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, args, conv3x3=common.default_conv, conv1x1=common.default_conv):
        super(MobileNet, self).__init__()
        self.width_mult = args.width_mult

        if args.data_train.find('CIFAR') >= 0:
            num_classes = int(args.data_train[5:])
        elif args.data_train.find('Tiny') >= 0:
            num_classes = 200
        else:
            num_classes = 1000

        stride = 1 if args.data_train.find('CIFAR') >= 0 else 2
        features = [ConvBNReLU(3, int(32 * self.width_mult), kernel_size=3, stride=stride)]
        features.extend(self._make_layers(in_planes=int(32 * self.width_mult)))
        self.features = nn.Sequential(*features)
        self.linear = nn.Sequential(nn.Dropout(0.2), nn.Linear(int(1024 * self.width_mult), num_classes))

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            out_planes = int(out_planes * self.width_mult)
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(DepthwiseSeparableConv(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = out.mean([2, 3])
        # embed()
        # out = F.avg_pool2d(out, 2)
        # out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = MobileNet()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

# test()
