import torch
import torch.nn as nn
from math import sqrt
from model import common


def make_model(args, parent=False):
    return DNCNN_FACTOR(args)


class conv_sic(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, norm=common.default_norm, act=common.default_act):
        super(conv_sic, self).__init__()
        self.channel_change = not in_channels == out_channels
        conv = [nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                          padding=kernel_size // 2, groups=in_channels, bias=bias),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=bias)]
        #if norm is not None: conv.append(norm(out_channels))
        self.body = nn.Sequential(*conv)
        if act is not None: self.act = nn.ReLU()


    def forward(self, x):
        res = self.body(x)
        x = self.act(res) if self.channel_change else self.act(res + x)
        return x


class conv_factor(nn.Module):
    def __init__(self, n_feats, kernel_size, sic_layer, stride=1, bias=True, norm=common.default_norm, act=common.default_act):
        super(conv_factor, self).__init__()
        body = [conv_sic(n_feats, n_feats, kernel_size, stride=stride, bias=bias, norm=norm, act=act) for _ in range(sic_layer)]
        body.append(norm(n_feats))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)


class BasicBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, factor_layer,
                 stride=1, bias=True, conv=conv_factor, norm=common.default_norm, act=common.default_act):
        super(BasicBlock, self).__init__()
        modules = [conv(n_feats, kernel_size, factor_layer, stride, bias, norm, act)]
        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        return self.conv(x)


class DNCNN_FACTOR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(DNCNN_FACTOR, self).__init__()

        n_blocks = args.m_blocks
        n_feats = args.n_feats
        kernel_size = 3 
        self.act = True
        bn  = args.bn
        sic_layer = args.sic_layer

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size), nn.ReLU(True)]

        # define body module
        m_body = [
            BasicBlock(
                n_feats, kernel_size, sic_layer
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
 
