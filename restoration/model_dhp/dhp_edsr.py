"""
This module is used to develop differentiable pruning via hypernetworks on EDSR for image denoising task.
"""

import torch
import torch.nn as nn
from model_dhp.dhp_base import DHP_Base, conv_dhp
from model import common
from model_dhp.dhp_srresnet import ResBlock_dhp, Upsampler

def make_model(args, parent=False):
    return EDSR(args)


class EDSR(DHP_Base):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__(args)
        width_mult = args.width_mult
        n_resblock = args.n_resblocks
        n_feats = int(args.n_feats * width_mult)
        scale = args.scale[0]

        kernel_size = 3
        act = nn.ReLU() # Pay attention to the difference between inplace and non-inplace operation

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.n_resblock = args.n_resblocks

        self.latent_vector_stage0 = nn.Parameter(torch.randn((3)))
        self.latent_vector_stage1 = nn.Parameter(torch.randn((n_feats)))

        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        # define head module
        self.head = conv_dhp(args.n_colors, n_feats, kernel_size, batchnorm=False, latent_vector=self.latent_vector_stage1, args=args)
        # define body module
        self.body = nn.ModuleList([ResBlock_dhp(n_feats, kernel_size, act=act, res_scale=args.res_scale,
                                                latent_vector=self.latent_vector_stage1, args=args)
                                   for _ in range(n_resblock)])
        self.body.append(conv_dhp(n_feats, n_feats, kernel_size, batchnorm=False, latent_vector=self.latent_vector_stage1, args=args))
        # define tail module
        if self.prune_upsampler:
            self.tail = nn.ModuleList([Upsampler(scale, n_feats, act=None, latent_vector=self.latent_vector_stage1, args=args),
                                       conv_dhp(n_feats, args.n_colors, kernel_size, batchnorm=False, args=args)])
        else:
            self.tail = nn.Sequential(*[common.Upsampler(conv, scale, n_feats),
                                        conv(n_feats, args.n_colors, 3)])
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.show_latent_vector()

    def mask(self):
        latent_vectors = self.gather_latent_vector(grad_prune=self.grad_prune, grad_normalize=self.grad_normalize)
        if self.prune_upsampler:
            masks = [torch.ones_like(latent_vectors[0], dtype=torch.bool, device='cuda')] + \
                    [v.abs() >= min(self.pt, v.abs().topk(self.mc)[0][-1]) for v in latent_vectors[1:-1]] + \
                    [torch.ones_like(latent_vectors[-1], dtype=torch.bool, device='cuda')]
        else:
            masks = [torch.ones_like(v, dtype=torch.bool, device='cuda') for v in latent_vectors[:2]] + \
                    [v.abs() >= min(self.pt, v.abs().topk(self.mc)[0][-1]) for v in latent_vectors[1:-1]]
        # mask has no gradients. Comparison operations are without gradients automatically.
        # when calculating the weights, biases, don't need to slice them?
        return masks

    def proximal_operator(self, lr):
        regularization = self.regularization * lr
        latent_vectors = self.gather_latent_vector()
        if self.prune_upsampler:
            latent_vectors = latent_vectors[1:-1]
        else:
            latent_vectors = latent_vectors[2:]
        for i, v in enumerate(latent_vectors):
            if torch.sum(v.abs() >= self.pt) > self.mc:
                self.soft_thresholding(v, regularization)

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
        x = self.sub_mean(x)
        if not self.finetuning:
            latent_vectors = self.gather_latent_vector()
            # for i, (k, v) in enumerate(zip(kk, latent_vectors)):
            #     print(i, list(v.shape), k)
            # embed()
            x = self.head([x, latent_vectors[0]])
            for i, layer in enumerate(self.body):
                if i == 0:
                    res = layer([x, latent_vectors[1]])
                else:
                    res = layer([x, latent_vectors[1]])
            if self.prune_upsampler:
                out = self.tail[0]([res + x, latent_vectors[1]])
                out = self.tail[1]([out, latent_vectors[1]])
            else:
                out = self.tail(res + x)
        else:
            x = self.head(x)
            for i, layer in enumerate(self.body):
                if i == 0:
                    res = layer(x)
                else:
                    res = layer(res)
            if self.prune_upsampler:
                out = self.tail[0](res + x)
                out = self.tail[1](out)
            else:
                out = self.tail(res + x)
        out = self.add_mean(out)
        # embed()
        return out

# for i, (k, v) in enumerate(zip(kk, latent_vectors)):
#     print(i, list(v.shape), k)