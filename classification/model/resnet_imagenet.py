"""
Author: Yawei Li
Date: 21/08/2019
Try to decompose resnet18 for ImageNet classification
"""

# from torchvision.models.resnet import ResNet, Bottleneck
# import torch.nn as nn
from torchvision import models
import torch.nn as nn
# from IPython import embed


def load_state_dict(self, state_dict, strict=False):
    """
    load state dictionary
    used to load the model parameters with different sizes during test
    """
    current_state = self.state_dict(keep_vars=True)
    for (current_name, current_param), (name, param) in zip(current_state.items(), state_dict.items()):
        # if name in own_state:
        if isinstance(param, nn.Parameter):
            param = param.data
        if param.size() != current_param.size():
            current_state[current_name].data = param
            print('Different shape {}'.format(param.shape))
        else:
            current_state[current_name].data.copy_(param)
            print('Same shape {}'.format(param.shape))


def set_channels(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            s = m.weight.shape
            m.in_channels, m.out_channels = s[1], s[0]
        elif isinstance(m, nn.BatchNorm2d):
            s = m.weight.shape
            m.num_features = s[0]
        elif isinstance(m, nn.Linear):
            s = m.weight.shape
            m.in_features, m.out_features = s[1], s[0]


def make_model(args, parent=False):
    net = 'resnet' + str(args[0].depth)
    net = getattr(models, net)(pretrained=False)
    net.load_state_dict = load_state_dict.__get__(net)
    return net


