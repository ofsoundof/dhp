import torch
import torch.nn as nn
from math import sqrt
from model import common


def make_model(args, parent=False):
    return DnCNN(args)


class Basic_Block(nn.Module):
    def __init__(self, conv, n_feat, kernel_size,bias=True, bn=False, act=True):
        super(Basic_Block, self).__init__()
        m = [conv(n_feat,n_feat,kernel_size,bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(n_feat))
        if act:
            m.append(nn.ReLU(True))
        self.body = nn.Sequential(*m)

    def forward(self, x):
        return self.body(x)


class DnCNN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(DnCNN, self).__init__()

        n_blocks = args.m_blocks
        n_feats = args.n_feats
        kernel_size = 3 
        self.act = True
        bn = args.bn

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size), nn.ReLU(True)]

        # define body module
        m_body = [
            Basic_Block(
                conv, n_feats, kernel_size, bn=bn, act=self.act
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
 
