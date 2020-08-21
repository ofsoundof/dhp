"""
This module is used to develop differentiable pruning via hypernetworks on DnCNN for image denoising task.
"""

import torch
import torch.nn as nn
from model_dhp.dhp_base import DHP_Base, conv_dhp


def make_model(args, parent=False):
    return DnCNN_DHP(args)


class DnCNN_DHP(DHP_Base):
    def __init__(self, args):
        super(DnCNN_DHP, self).__init__(args)
        self.width_mult = args.width_mult
        n_blocks = args.m_blocks
        n_feats = int(args.n_feats * self.width_mult)
        kernel_size = 3
        # act = nn.ReLU()

        self.latent_vector = nn.Parameter(torch.randn((1)))
        # define head module
        self.head = conv_dhp(args.n_colors, n_feats, kernel_size, batchnorm=False, args=args)
        # define body module
        self.body = nn.ModuleList([conv_dhp(n_feats, n_feats, kernel_size, act=nn.ReLU(), args=args) for _ in range(n_blocks)])
        # define tail module
        self.tail = conv_dhp(n_feats, args.n_colors, kernel_size, batchnorm=False, args=args)
        self.show_latent_vector()

    def mask(self):
        latent_vectors = self.gather_latent_vector(grad_prune=self.grad_prune, grad_normalize=self.grad_normalize)
        l = len(latent_vectors)
        masks = []
        for i, v in enumerate(latent_vectors):
            channels = self.remain_channels(v)
            if i == 0 or i == l - 1:
                masks.append(torch.ones_like(v, dtype=torch.bool, device='cuda'))
            else:
                masks.append(v.abs() >= min(self.pt, v.abs().topk(channels)[0][-1]))

        # mask has no gradients. Comparison operations are without gradients automatically.
        # when calculating the weights, biases, don't need to slice them?
        return masks

    def proximal_operator(self, lr):
        regularization = self.regularization * lr
        latent_vectors = self.gather_latent_vector()
        l = len(latent_vectors)
        for i, v in enumerate(latent_vectors):
            channels = self.remain_channels(v)
            if i != 0 and i != l - 1:
                if torch.sum(v.abs() >= self.pt) > channels:
                    self.soft_thresholding(v, regularization)

    def set_parameters(self, calc_weight=False):
        latent_vectors, masks = self.gather_latent_vector(), self.mask()
        self.head.set_parameters([latent_vectors[0]] + masks[:2], calc_weight)
        for i, layer in enumerate(self.body):
            layer.set_parameters([latent_vectors[i + 1]] + masks[i + 1: i + 3], calc_weight)
        self.tail.set_parameters([latent_vectors[16]] + masks[16:], calc_weight)

    def forward(self, x):
        if not self.finetuning:
            latent_vectors = self.gather_latent_vector()
            residual = self.head([x, latent_vectors[0]])
            for i, layer in enumerate(self.body):
                residual = layer([residual, latent_vectors[i + 1]])
            residual = self.tail([residual, latent_vectors[16]])
        else:
            residual = self.head(x)
            for i, layer in enumerate(self.body):
                residual = layer(residual)
            residual = self.tail(residual)
            # residual = self.tail(self.body(self.head(x)))
        return residual + x

# for i, (k, v) in enumerate(zip(kk, latent_vectors)):
#     print(i, list(v.shape), k)