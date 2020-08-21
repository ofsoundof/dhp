import torch.nn as nn
from model import common
import torch.nn.functional as F
from IPython import embed

def make_model(args, parent=False):
    if args[0].weight_type.find('original') >= 0:
        conv3x3 = common.default_conv
    elif args[0].weight_type.find('single') >= 0:
        conv3x3 = common.ACSConv
    else:
        conv3x3 = common.ACConv
    return CFQKBN(args[0], conv3x3)

class CFQKBN(nn.Module):

    def __init__(self, args, conv3x3=common.default_conv):
        super(CFQKBN, self).__init__()
        n_classes = 10 if args.data_train == 'CIFAR10' else 100
        self.conv1 = common.BasicBlock(3, 32, 5, stride=1, bias=False, conv3x3=conv3x3, args=args)
        self.conv2 = common.BasicBlock(32, 32, 5, stride=1, bias=False, conv3x3=conv3x3, args=args)
        self.conv3 = common.BasicBlock(32, 64, 5, stride=1, bias=False, conv3x3=conv3x3, args=args)
        self.fc1 = nn.Linear(in_features=3 * 3 * 64, out_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=64, out_features=n_classes)

    def forward(self, x):
        x = self.conv1(x)   #   32
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0)   #15
        x = self.conv2(x)
        x = F.avg_pool2d(x, kernel_size=3, stride=2, padding=0)   #7
        x = self.conv3(x)
        x = F.avg_pool2d(x, kernel_size=3, stride=2, padding=0)   #3
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x