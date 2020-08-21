"""
Author: Yawei Li
Date: 20/08/2019
Factor method for ICCV2019 paper.
"""

import torch.nn as nn
from model import common


def make_model(args, parent=False):
    return SRResNet_Factor(args)


class SRResNet_Factor(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SRResNet_Factor, self).__init__()

        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        scale = args.scale[0]

        kernel_size = 3
        act = nn.PReLU()

        head = [conv(args.n_colors, n_feats, kernel_size=9), act]
        body = [common.ResBlock_Factor(common.conv_factor, n_feats, kernel_size, bn=True, act=act,
                                       sic_layer=args.sic_layer) for _ in range(n_resblock)]
        body.extend([conv(n_feats, n_feats, kernel_size), nn.BatchNorm2d(n_feats)])

        tail = [
            common.Upsampler(conv, scale, n_feats, act=act),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        x = self.head(x)
        f = self.body(x)
        x = self.tail(x + f)
        return x