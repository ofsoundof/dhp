import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, stride=1, groups=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride=stride,
        padding=(kernel_size//2), groups=groups, bias=bias)

def default_norm(in_channels):
    return nn.BatchNorm2d(in_channels)

def default_act():
    return nn.ReLU(True)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False, bn=True, act=nn.ReLU(True)):
        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=False, bn=False, act='relu', res_scale=1, num_conv=2):
        super(ResBlock, self).__init__()
        m = []
        for i in range(num_conv):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU())

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(nn.PReLU())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(nn.PReLU())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


########################################################################################################################
# SIC layer of Factorized convlutional neural networks
########################################################################################################################
class conv_sic(nn.Module):
    def __init__(self, n_feat, kernel_size, act):
        super(conv_sic, self).__init__()
        conv = [nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size//2, groups=n_feat),
                nn.Conv2d(n_feat, n_feat, 1)]
        self.body = nn.Sequential(*conv)
        self.act = act

    def forward(self, x):
        y = self.body(x)
        return self.act(y + x)


########################################################################################################################
# Compressed conv layer of Factor
########################################################################################################################
class conv_factor(nn.Module):
    def __init__(self, n_feat, kernel_size, sic_layer=2, act=nn.ReLU()):
        super(conv_factor, self).__init__()
        body = [conv_sic(n_feat, kernel_size, act) for i in range(sic_layer)]
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)


########################################################################################################################
# Compressed ResBlock of Factor
########################################################################################################################
class ResBlock_Factor(nn.Module):
    def __init__(
        self, conv=conv_factor, n_feat=64, kernel_size=3,
            bn=False, act=nn.ReLU(True), res_scale=1, num_conv=2, sic_layer=2):
        super(ResBlock_Factor, self).__init__()
        m = []
        for i in range(num_conv):
            m.append(conv(n_feat, kernel_size, sic_layer=sic_layer, act=act))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


########################################################################################################################
# Compressed conv layer of Group
########################################################################################################################
class conv_group(nn.Module):
    def __init__(self, n_feat, kernel_size, group_size):
        super(conv_group, self).__init__()
        groups = n_feat // group_size
        modules = [default_conv(n_feat, n_feat, kernel_size=kernel_size, stride=1, groups=groups)]
        modules.append(default_conv(n_feat, n_feat, kernel_size=1, stride=1))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        return self.body(x)


########################################################################################################################
# Compressed ResBlock of Group
########################################################################################################################
class ResBlock_Group(nn.Module):
    def __init__(
        self, conv=conv_group, n_feat=64, kernel_size=3,
            bn=False, act=nn.ReLU(True), res_scale=1, num_conv=2, group_size=64):
        super(ResBlock_Group, self).__init__()

        m = []
        for i in range(num_conv):
            m.append(conv(n_feat, kernel_size, group_size=group_size))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


########################################################################################################################
# Used by conv_basis_bn_every
########################################################################################################################
class conv_one(nn.Module):
    def __init__(self, basis, n_feat, kernel_size, basis_size, n_basis):
        super(conv_one, self).__init__()
        self.n_feat = n_feat
        self.kernel_size = kernel_size
        self.basis_size = basis_size
        self.n_basis = n_basis
        self.group = n_feat // basis_size

        self.basis_weight = basis
        self.basis_bias = nn.Parameter(torch.zeros(n_basis)) #if bias else None

    def forward(self, x):
        if self.group == 1:
            x = F.conv2d(input=x, weight=self.basis_weight, bias=self.basis_bias, padding=self.kernel_size//2)
        else:
            x = torch.cat([F.conv2d(input=xi, weight=self.basis_weight, bias=self.basis_bias, padding=self.kernel_size//2)
                           for xi in torch.split(x, self.basis_size, dim=1)], dim=1)
        return x

    def __repr__(self):
        s = 'Conv_one(in_channels={}, basis_size={}, n_basis={}, group[{}]=in_channels[{}] / basis_size[{}], ' \
            'out_channels[{}]=n_basis[{}] x group[{}])'\
            .format(self.n_feat, self.basis_size, self.n_basis, self.group, self.n_feat, self.basis_size,
                    self.n_basis * self.group, self.n_basis, self.group)
        return s


########################################################################################################################
# Originally used for SRResNet. BatchNorm and Activation is used after every conv layer no matter it's a 3x3 conv or
# a 1x1 projection
########################################################################################################################
class conv_basis_bn_every(nn.Module):
    def __init__(self, basis, n_feat, n_basis, basis_size, kernel_size, bn=True, act=nn.PReLU()):
        super(conv_basis_bn_every, self).__init__()
        group = n_feat // basis_size
        conv = default_conv

        m = [conv_one(basis, n_feat, kernel_size, basis_size, n_basis)]
        if bn: m.append(nn.BatchNorm2d(group * n_basis))
        if act is not None: m.append(act)
        m.append(conv(group * n_basis, n_feat, kernel_size=1))
        if bn: m.append(nn.BatchNorm2d(n_feat))
        if act is not None: m.append(act)
        self.conv = nn.Sequential(*m)

    def forward(self, x):
        return self.conv(x)


########################################################################################################################
# Originally used for EDSR. If BatchNorm and Activation are needed, then they are only appended to the 1x1 projection.
########################################################################################################################
class conv_basis(nn.Module):
    def __init__(self, basis, n_feat, n_basis, basis_size, kernel_size):
        super(conv_basis, self).__init__()
        self.n_feat = n_feat
        self.kernel_size = kernel_size
        self.basis_size = basis_size
        self.n_basis = n_basis
        self.group = n_feat // basis_size

        self.basis_weight = basis
        self.basis_bias = nn.Parameter(torch.zeros(n_basis)) #if bias else None
        self.proj_weight = nn.Parameter(nn.init.kaiming_uniform_(torch.ones(n_feat, n_basis * self.group, 1, 1)))
        self.proj_bias = nn.Parameter(torch.zeros(n_feat)) #if bias else None

    def __repr__(self):
        s = 'Conv_basis(in_channels={}, basis_size={}, n_basis={}, group={}, out_channels={})\n' \
            '     Conv1(in_channels={}, out_channels[{}]=n_basis[{}] x in_channels[{}] / basis_size[{}])\n' \
            '     Conv2(in_channels={}, out_channels={})'\
            .format(self.n_feat, self.basis_size, self.group, self.n_basis, self.n_feat, self.n_feat,
                    self.n_basis * self.group, self.n_basis, self.n_feat, self.basis_size, self.n_basis * self.group, self.n_feat)
        return s

    def forward(self, x):
        if self.group == 1:
            return self.forward_one_group(x)
        else:
            return self.forward_multi_group(x)

    def forward_one_group(self, x):
        x = F.conv2d(input=x, weight=self.basis_weight, bias=self.basis_bias, padding=self.kernel_size//2)
        x = F.conv2d(input=x, weight=self.proj_weight, bias=self.proj_bias, padding=0)
        return x

    def forward_multi_group(self, x):
        '''
        When basis_size=64, use F.relu, CUDA out of memory. Without F.relu, problem sovlved.
        When basis_size=32 and without F.relu, still CUDA out of memory.
        The memory bottle neck may be torch.split and torch.cat.
        '''
        x = torch.cat([F.conv2d(input=xi, weight=self.basis_weight, bias=self.basis_bias, padding=self.kernel_size//2)
                       for xi in torch.split(x, self.basis_size, dim=1)],dim=1)
        x = F.conv2d(input=x, weight=self.proj_weight, bias=self.proj_bias, padding=0)
        return x

    def forward_multi(self, x):
        x = torch.unsqueeze(x, 1)
        weight = torch.unsqueeze(self.basis_weight, 1)
        x = F.conv3d(input=x, weight=weight, bias=self.basis_bias, stride=[self.basis_size, 1, 1],
                     padding=[0, self.kernel_size//2, self.kernel_size//2])
        shape = x.shape
        x = torch.reshape(x, [shape[0], shape[1]*shape[2], shape[3], shape[4]])
        x = F.conv2d(input=x, weight=self.proj_weight, bias=self.proj_bias, padding=0)
        return x


########################################################################################################################
# ResBlock of the network compressed by our basis learning method.
# Use conv_basis_bn_every for SRResNet. bn_every is set to true. The batchnorm layer is inserted directly after every conv.
# Use conv_basis for EDSR.
########################################################################################################################
class ResBlock_Basis(nn.Module):
    def __init__(self, n_feat, kernel_size, basis_size, n_basis, share_basis, conv=conv_basis_bn_every,
                 bn=False, act=nn.ReLU(True), res_scale=1, bn_every=False):
        super(ResBlock_Basis, self).__init__()

        decom_basis1 = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(n_basis, basis_size, kernel_size, kernel_size)))
        if share_basis:
            decom_basis2 = decom_basis1
        else:
            decom_basis2 = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(n_basis, basis_size, kernel_size, kernel_size)))
        m = []
        if bn_every:
            m.extend([conv(decom_basis1, n_feat, n_basis, basis_size, kernel_size, bn, act),
                      conv(decom_basis2, n_feat, n_basis, basis_size, kernel_size, bn, act)])
        else:
            m.append(conv(decom_basis1, n_feat, n_basis, basis_size, kernel_size))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            m.extend([act, conv(decom_basis2, n_feat, n_basis, basis_size, kernel_size)])
            if bn: m.append(nn.BatchNorm2d(n_feat))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


def act_vconv(res_act):
    res_act = res_act.lower()
    if res_act == 'softplus':
        act = nn.Softplus()
    elif res_act == 'sigmoid':
        act = nn.Sigmoid()
    elif res_act == 'tanh':
        act = nn.Tanh()
    elif res_act == 'elu':
        act = nn.ELU()
    else:
        raise NotImplementedError
    return act