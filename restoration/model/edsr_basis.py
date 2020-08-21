"""
Author: Yawei Li
Date: 20/08/2019
Basis learning method applied to EDSR for ICCV2019 paper.
"""

from model import common
import torch.nn as nn
import torch


def make_model(args, parent=False):
    return EDSR_Basis(args)


class EDSR_Basis(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR_Basis, self).__init__()

        basis_size = args.basis_size
        n_basis = args.n_basis
        share_basis = args.share_basis
        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        scale = args.scale[0]
        bn_every = args.bn_every

        kernel_size = 3
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]
        # define body module
        m_body = [common.ResBlock_Basis(n_feats, kernel_size, basis_size, n_basis, share_basis, conv=common.conv_basis,
                                        act=act, res_scale=args.res_scale, bn_every=bn_every) for _ in range(n_resblock)]
        m_body.append(conv(n_feats, n_feats, kernel_size))
        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        # from IPython import embed; embed()
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        x = self.tail(res + x)
        x = self.add_mean(x)

        return x

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
    for b in range(n_resblock):

        pretrain_weight1 = pre_values[b * 4 + 6]
        pretrain_weight2 = pre_values[b * 4 + 8]

        if not share_basis:
            step = 2 if bias else 1
            basis1 = current_state[b * 4 * step + 6]
            coordinate1 = current_state[b * 4 * step + 6 + step]
            basis2 = current_state[b * 4 * step + 6 + step * 2]
            coordinate2 = current_state[b * 4 * step + 6 + step * 3]
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

# def loss_weight_difference(model, args, pretrain_state, para_loss_type='L2'):
#     if args.loss_norm_default:
#         return loss_norm_default(model, args, pretrain_state, para_loss_type)
#     else:
#         return loss_norm(model, args, pretrain_state, para_loss_type)
