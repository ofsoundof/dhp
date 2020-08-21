"""
Author: Yawei Li
Date: 20/08/2019
Basis learning method applied to SRResNet for ICCV2019 paper.
"""

import torch.nn as nn
import torch
from model import common


def make_model(args, parent=False):
    return SRResNet_Basis(args)


class SRResNet_Basis(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SRResNet_Basis, self).__init__()

        basis_size = args.basis_size
        n_basis = args.n_basis
        share_basis = args.share_basis
        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        scale = args.scale[0]
        bn_every = args.bn_every

        kernel_size = 3
        act = nn.PReLU()

        head = [conv(args.n_colors, n_feats, kernel_size=9), act]
        body = [common.ResBlock_Basis(n_feats, kernel_size, basis_size, n_basis, share_basis, bn=True, act=act,
                                      bn_every=bn_every) for _ in range(n_resblock)]
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


def loss_type(loss_para_type):
    if loss_para_type == 'L1':
        loss_fun = nn.L1Loss()
    elif loss_para_type == 'L2':
        loss_fun = nn.MSELoss()
    else:
        raise NotImplementedError
    return loss_fun

def form_weight(args, basis, coordinate):
    n_basis = args.n_basis
    n_feats = args.n_feats
    basis_size = args.basis_size
    group = n_feats // basis_size
    kernel_size = 3
    basis = torch.transpose(torch.reshape(basis, [n_basis, -1]), 0, 1)
    coordinate = torch.transpose(torch.reshape(torch.squeeze(coordinate), [n_feats * group, n_basis]), 0, 1)
    weight = torch.mm(basis, coordinate)
    weight = torch.reshape(torch.transpose(weight, 0, 1), [n_feats, n_feats, kernel_size, kernel_size])
    # coordinate = torch.reshape(torch.reshape(torch.transpose(torch.squeeze(coordinate), 0, 1),
    #                                 [n_basis, group, n_feats]),
    #                                 [n_basis, group * n_feats])
    # weight = torch.mm(basis, coordinate)
    # weight = torch.reshape(weight,
    #                        [basis_size, kernel_size, kernel_size, group, n_feats])
    # weight = torch.reshape(weight.permute(4, 3, 0, 1, 2),
    #                        [n_feats, n_feats, kernel_size, kernel_size])
    return weight
#
#
# def loss_norm_default(model, args, pretrain_state, para_loss_type='L2'):
#     loss = torch.tensor([0.0])
#     norm = torch.tensor([0.0])
#     return loss.to('cuda'), norm.to('cuda')

def loss_norm_difference(model, args, pretrain_state, para_loss_type='L2'):
    #from IPython import embed; embed(); exit()
    n_resblock = args.n_resblocks
    share_basis = args.share_basis
    bias = args.edsr_de_conv_bias

    current_state = list(model.parameters())
    pre_keys = list(pretrain_state.keys())
    pre_values = [v for _, v in pretrain_state.items()]
    loss_fun = loss_type(para_loss_type)
    loss = 0
    norm = 0
    #from IPython import embed; embed(); exit()
    for b in range(n_resblock):

        pretrain_weight1 = pre_values[b * 15 + 3]
        pretrain_weight2 = pre_values[b * 15 + 11]

        if not share_basis:
            basis1 = current_state[b * 20 + 3]
            coordinate1 = current_state[b * 20 + 8]
            basis2 = current_state[b * 20 + 13]
            coordinate2 = current_state[b * 20 + 18]
        else:
            if bias:
                basis1 = current_state[b * 7 + 6]
                coordinate1 = current_state[b * 7 + 8]
                basis2 = current_state[b * 7 + 6]
                coordinate2 = current_state[b * 7 + 11]
            else:
                basis1 = current_state[b * 3 + 6]
                coordinate1 = current_state[b * 3 + 7]
                basis2 = current_state[b * 3 + 6]
                coordinate2 = current_state[b * 3 + 8]


        # print the keys
        # print('pretrain_weight: {}, current_basis: {}, current_coordinate: {}'.
        #       format(key, current_state_dict[(index-6)//2*3+6], current_state_dict[(index-6)//2*3+7]))
        # from IPython import embed; embed()
        current_weight1 = form_weight(args, basis1, coordinate1)
        current_weight2 = form_weight(args, basis2, coordinate2)

        loss = loss + loss_fun(pretrain_weight1, current_weight1) + loss_fun(pretrain_weight2, current_weight2)
        norm += torch.mean(basis1 ** 2)
    if share_basis:
        norm += torch.mean(basis2 ** 2)

    return loss, norm
