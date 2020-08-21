"""
Developing differential pruning via hypernetworks for SRResnet
"""

import torch.nn as nn
import torch
from model import common
from model_dhp.dhp_base import conv_dhp, DHP_Base
# from IPython import embed


def make_model(args, parent=False):
    return SRResNet_DHP(args)


class ResBlock_dhp(nn.Module):
    def __init__(self, n_feat, kernel_size, bias=True, bn=False, act=nn.PReLU(), res_scale=1, latent_vector=None, args=None):
        super(ResBlock_dhp, self).__init__()
        self.finetuning = False
        self.res_scale = res_scale
        self.layer1 = conv_dhp(n_feat, n_feat, kernel_size, bias=bias, batchnorm=bn, act=act, args=args)
        self.layer2 = conv_dhp(n_feat, n_feat, kernel_size, bias=bias, batchnorm=bn, act=None, latent_vector=latent_vector, args=args)

        self.res_scale = res_scale

    def set_parameters(self, vector_mask, calc_weight=False):
        self.layer1.set_parameters(vector_mask[:3], calc_weight)
        self.layer2.set_parameters([self.layer1.latent_vector] + vector_mask[2:0:-1], calc_weight)

    def forward(self, x):
        if not self.finetuning:
            x, latent_vector = x
            res = self.layer1([x, latent_vector])
            res = self.layer2([res, self.layer1.latent_vector])
        else:
            res = self.layer2(self.layer1(x))
        res = res * self.res_scale
        res += x
        return res


class Upsampler(nn.Module):
    def __init__(self, scale, n_feat, bn=False, act=nn.PReLU(), bias=True, latent_vector=None, args=None):
        super(Upsampler, self).__init__()
        self.finetuning = False
        self.scale = scale
        if self.scale == 4: scale = 2
        self.upsampler1 = conv_dhp(n_feat, scale ** 2 * n_feat, 3, bias=bias, batchnorm=bn,latent_vector=latent_vector, args=args, scale=scale)
        self.pixel_shuffler1 = nn.PixelShuffle(scale)
        if act is not None: self.relu1 = act
        if self.scale == 4:
            self.upsampler2 = conv_dhp(n_feat, scale ** 2 * n_feat, 3, bias=bias, batchnorm=bn, latent_vector=latent_vector, args=args, scale=scale)
            self.pixel_shuffler2 = nn.PixelShuffle(scale)
            if act is not None: self.relu2 = act
        else:
            raise NotImplementedError('Upsampling scale {} is not implemented.'.format(self.scale))

    def set_parameters(self, vector_mask, calc_weight=False):
        scale = 2 if self.scale == 4 else self.scale
        vector_mask_output = [v.repeat_interleave(scale ** 2) for v in vector_mask]
        self.upsampler1.set_parameters(vector_mask + vector_mask_output, calc_weight)
        if hasattr(self, 'relu1'):
            self.relu1.channels = self.upsampler1.in_channels_remain if hasattr(self.upsampler1, 'in_channels_remain') else self.upsampler1.in_channels
        if self.scale == 4:
            self.upsampler2.set_parameters(vector_mask + vector_mask_output, calc_weight)
            if hasattr(self, 'relu2'):
                self.relu2.channels = self.upsampler2.in_channels_remain if hasattr(self.upsampler2, 'in_channels_remain') else self.upsampler2.in_channels

    def forward(self, x):
        if hasattr(self, 'relu1') or hasattr(self, 'relu2'):
            if not self.finetuning:
                x, latent_vector = x
                x = self.pixel_shuffler1(self.relu1(self.upsampler1([x, latent_vector])))
                if self.scale == 4:
                    x = self.pixel_shuffler2(self.relu2(self.upsampler2([x, latent_vector])))
            else:
                x = self.pixel_shuffler1(self.relu1(self.upsampler1(x)))
                if self.scale == 4:
                    x = self.pixel_shuffler2(self.relu2(self.upsampler2(x)))
        else:
            if not self.finetuning:
                x, latent_vector = x
                x = self.pixel_shuffler1(self.upsampler1([x, latent_vector]))
                if self.scale == 4:
                    x = self.pixel_shuffler2(self.upsampler2([x, latent_vector]))
            else:
                x = self.pixel_shuffler1(self.upsampler1(x))
                if self.scale == 4:
                    x = self.pixel_shuffler2(self.upsampler2(x))
        return x


class SRResNet_DHP(DHP_Base):
    def __init__(self, args, conv=common.default_conv):
        super(SRResNet_DHP, self).__init__(args)
        self.width_mult = args.width_mult
        self.n_resblock = args.n_resblocks
        self.n_feats = int(args.n_feats * self.width_mult)
        self.scale = args.scale[0]
        # act = nn.PReLU()

        self.latent_vector_stage0 = nn.Parameter(torch.randn((3)))
        self.latent_vector_stage1 = nn.Parameter(torch.randn((self.n_feats)))

        self.head = conv_dhp(self.n_colors, self.n_feats, 9, batchnorm=False, act=nn.PReLU(), latent_vector=self.latent_vector_stage1, args=args)
        self.body = nn.ModuleList([ResBlock_dhp(self.n_feats, 3, bn=True, act=nn.PReLU(), latent_vector=self.latent_vector_stage1, args=args) for _ in range(self.n_resblock)])
        self.body.append(conv_dhp(self.n_feats, self.n_feats, 3, latent_vector=self.latent_vector_stage1, args=args))

        if self.prune_upsampler:
            self.tail = nn.ModuleList([Upsampler(self.scale, self.n_feats, act=nn.PReLU(), latent_vector=self.latent_vector_stage1, args=args)])
            self.tail.append(conv_dhp(self.n_feats, self.n_colors, 3, batchnorm=False, args=args))
        else:
            self.tail = nn.Sequential(*[common.Upsampler(conv, self.scale, self.n_feats, act=nn.PReLU()),
                                        conv(self.n_feats, self.n_colors, 3)])
        self.show_latent_vector()

    def mask(self):
        latent_vectors = self.gather_latent_vector(grad_prune=self.grad_prune, grad_normalize=self.grad_normalize)
        l = len(latent_vectors)
        masks = []
        if self.prune_upsampler:
            flag = [0, l - 1]
        else:
            flag = [0, 1]
        for i, v in enumerate(latent_vectors):
            channels = self.remain_channels(v)
            if i in flag:
                masks.append(torch.ones_like(v, dtype=torch.bool, device='cuda'))
            else:
                masks.append(v.abs() >= min(self.pt, v.abs().topk(channels)[0][-1]))
        # if self.prune_upsampler:
        #     masks = [torch.ones_like(latent_vectors[0], dtype=torch.bool, device='cuda')] + \
        #             [v.abs() >= min(self.pt, v.abs().topk(self.mc)[0][-1]) for v in latent_vectors[1:-1]] + \
        #             [torch.ones_like(latent_vectors[-1], dtype=torch.bool, device='cuda')]
        # else:
        #     masks = [torch.ones_like(v, dtype=torch.bool, device='cuda') for v in latent_vectors[:2]] + \
        #             [v.abs() >= min(self.pt, v.abs().topk(self.mc)[0][-1]) for v in latent_vectors[1:-1]]
        # mask has no gradients. Comparison operations are without gradients automatically.
        # when calculating the weights, biases, don't need to slice them?
        return masks

    def proximal_operator(self, lr):
        regularization = self.regularization * lr
        latent_vectors = self.gather_latent_vector()
        l = len(latent_vectors)
        if self.prune_upsampler:
            flag = [0, l - 1]
        else:
            flag = [0, 1]
        for i, v in enumerate(latent_vectors):
            channels = self.remain_channels(v)
            if i not in flag:
                if torch.sum(v.abs() >= self.pt) > channels:
                    self.soft_thresholding(v, regularization)
        # if self.prune_upsampler:
        #     latent_vectors = latent_vectors[1:-1]
        # else:
        #     latent_vectors = latent_vectors[2:]
        # for i, v in enumerate(latent_vectors):
        #     # embed()
        #     if torch.sum(v.abs() >= self.pt) > self.mc:
        #         self.soft_thresholding(v, regularization)

    def set_parameters(self, calc_weight=False):
        latent_vectors, masks = self.gather_latent_vector(), self.mask()
        # former_vectors, last_vectors, other_vectors, former_masks, last_masks, other_masks = self.mask()

        self.head.set_parameters([latent_vectors[0]] + masks[0:2], calc_weight)
        for i, layer in enumerate(self.body):
            if i != self.n_resblock:
                vm = [latent_vectors[1], masks[1], masks[i + 2]]
            else:
                vm = [latent_vectors[1], masks[1], masks[1]]
            layer.set_parameters(vm, calc_weight)
        if self.prune_upsampler:
            self.tail[0].set_parameters([latent_vectors[1], masks[1]], calc_weight)
            self.tail[1].set_parameters([latent_vectors[1], masks[1], masks[-1]], calc_weight)

    def forward(self, x):
        latent_vectors = self.gather_latent_vector()
        # for i, (k, v) in enumerate(zip(kk, latent_vectors)):
        #     print(i, list(v.shape), k)
        # embed()
        if not self.finetuning:
            x = self.head([x, latent_vectors[0]])
            for i, layer in enumerate(self.body):
                if i == 0:
                    f = layer([x, latent_vectors[1]])
                else:
                    f = layer([f, latent_vectors[1]])
            if self.prune_upsampler:
                f = self.tail[0]([f + x, latent_vectors[1]])
                f = self.tail[1]([f, latent_vectors[1]])
            else:
                f = self.tail(x + f)
        else:
            x = self.head(x)
            for i, layer in enumerate(self.body):
                if i == 0:
                    f = layer(x)
                else:
                    f = layer(f)
            if self.prune_upsampler:
                f = self.tail[0](x + f)
                f = self.tail[1](f)
            else:
                f = self.tail(x + f)
        # embed()
        return f

# for i, (k, v) in enumerate(zip(kk, latent_vectors)):
#     print(i, list(v.shape), k)
