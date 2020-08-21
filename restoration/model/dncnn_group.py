import torch
import torch.nn as nn
from math import sqrt
from model import common
import torch.nn.functional as F


def make_model(args, parent=False):
    return DNCNN_GROUP(args)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, group_size, kernel_size,
                 stride=1, bias=True, conv=common.default_conv, norm=common.default_norm, act=common.default_act):
        super(BasicBlock, self).__init__()
        groups = in_channels // group_size
        modules = [conv(in_channels, in_channels, kernel_size=kernel_size, stride=stride, bias=bias, groups=groups)]
        modules.append(conv(in_channels, out_channels, kernel_size=1, bias=bias))
        if norm is not None: modules.append(norm(out_channels))
        if act is not None: modules.append(act())
        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        return self.conv(x)

class DNCNN_GROUP(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(DNCNN_GROUP, self).__init__()

        n_blocks = args.m_blocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale
        self.act = True
        group_size = args.group_size
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size), nn.ReLU(True)]

        # define body module
        m_body = [
            BasicBlock(
                n_feats, n_feats, group_size, kernel_size
            ) for _ in range(n_blocks)
        ]

        # define tail module
        m_tail = [conv(n_feats, args.n_colors, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        residual = self.head(x)
        out = self.body(residual)


        return self.tail(out)+x
