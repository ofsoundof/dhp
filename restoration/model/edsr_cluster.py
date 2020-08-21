"""
Author: Yawei Li
Date: 20/08/2019
Clustering convolutional kernels. ECCV2018
"""

from model import common
import torch.nn as nn
import torch
from IPython import embed

def make_model(args, parent=False):
    return EDSR_CLUSTER(args)

# class my_conv(nn.Module):
#     def __init__(self, in_feat, out_feat, kernel_size, stride=1, bias=False):
#         super(my_conv, self).__init__()
#         body = [nn.Conv2d(in_feat, 64, kernel_size, padding=kernel_size//2, bias=bias),
#                 nn.Conv2d(64, out_feat, 1, padding=0, bias=bias)]
#         self.body = nn.Sequential(*body)
#
#     def forward(self, x):
#         return self.body(x)


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1, num_conv=2):

        super(ResBlock, self).__init__()
        m = []
        for i in range(num_conv):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class EDSR_CLUSTER(nn.Module):
    def __init__(self, args, conv3x3=common.default_conv, conv1x1=common.default_conv):
        super(EDSR_CLUSTER, self).__init__()

        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        self.scale = args.scale[0]
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)

        # define head module
        m_head = [conv3x3(args.n_colors, n_feats, kernel_size)]
        # define body module
        m_body = [
            ResBlock(
                conv3x3, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblock)
        ]
        m_body.append(conv3x3(n_feats, n_feats, kernel_size))
        # define tail module
        m_tail = [
            common.Upsampler(conv3x3, self.scale, n_feats, act=False),
            conv3x3(n_feats, args.n_colors, kernel_size)
        ]

        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        # embed()
        if conv3x3 == common.default_conv:
            self.load_state_dict(torch.load(args.pretrain_cluster))

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x


    # def load_state_dict(self, state_dict, strict=True):
    #     own_state = self.state_dict()
    #     for name, param in state_dict.items():
    #         if name in own_state:
    #             if isinstance(param, nn.Parameter):
    #                 param = param.data
    #             try:
    #                 own_state[name].copy_(param)
    #             except Exception:
    #                 if name.find('tail') == -1:
    #                     raise RuntimeError('While copying the parameter named {}, '
    #                                        'whose dimensions in the model are {} and '
    #                                        'whose dimensions in the checkpoint are {}.'
    #                                        .format(name, own_state[name].size(), param.size()))
    #         elif strict:
    #             if name.find('tail') == -1:
    #                 raise KeyError('unexpected key "{}" in state_dict'
    #                                .format(name))
    #
    #
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

