"""
This module is used to develop differentiable pruning via hypernetworks on unet for image denoising task.
"""

import torch
import torch.nn as nn
from model_dhp.dhp_base import conv_dhp, DHP_Base
from torch.nn import init
import os


def make_model(args, parent=False):
    return UNet_DHP(args)


class DownConv_dhp(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True, args=None):
        super(DownConv_dhp, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.finetuning = False

        self.conv1 = conv_dhp(self.in_channels, self.out_channels, 3, batchnorm=False, act=nn.ReLU(), args=args)
        self.conv2 = conv_dhp(self.out_channels, self.out_channels, 3, batchnorm=False, act=nn.ReLU(), args=args)

        # self.conv1 = conv3x3(self.in_channels, self.out_channels)
        # self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def set_parameters(self, vector_mask, calc_weight=False):
        self.conv1.set_parameters(vector_mask[:3], calc_weight)
        self.conv2.set_parameters([self.conv1.latent_vector] + vector_mask[2:], calc_weight)

    def forward(self, x):
        if not self.finetuning:
            x, latent_vector = x
            x = self.conv1([x, latent_vector])
            x = self.conv2([x, self.conv1.latent_vector])
        else:
            x = self.conv1(x)
            x = self.conv2(x)
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv_dhp(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, args=None):
        super(UpConv_dhp, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.finetuning = False
        # self.merge_mode = merge_mode
        # self.up_mode = up_mode
        # self.flag = flag

        # self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)
        self.upconv = conv_dhp(self.in_channels, self.out_channels, 2, stride=2, batchnorm=False, args=args, transpose=True)
        self.conv1 = conv_dhp(2 * self.out_channels, self.out_channels, 3, batchnorm=False, act=nn.ReLU(), args=args)
        self.conv2 = conv_dhp(self.out_channels, self.out_channels, 3, batchnorm=False, act=nn.ReLU(), args=args)
        # if self.merge_mode == 'concat' and self.flag:
        #     self.conv1 = conv3x3(2*self.out_channels, self.out_channels)
        # else:
        #     # num of input channels to conv2 is same
        #     self.conv1 = conv3x3(self.out_channels, self.out_channels)
        # self.conv2 = conv3x3(self.out_channels, self.out_channels)

    def set_parameters(self, vector_mask, calc_weight=False):
        self.upconv.set_parameters(vector_mask[:3], calc_weight)
        vector = torch.cat([self.upconv.latent_vector, vector_mask[-2]], 0)
        mask = torch.cat(vector_mask[2::4], 0)
        self.conv1.set_parameters([vector, mask, vector_mask[3]], calc_weight)
        self.conv2.set_parameters([self.conv1.latent_vector] + vector_mask[3:5], calc_weight)

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        if not self.finetuning:
            from_up, latent_vector_conv, latent_vector_conv_transpose = from_up
            # embed()
            from_up = self.upconv([from_up, latent_vector_conv])
            x = torch.cat([from_up, from_down], 1)
            x = self.conv1([x, torch.cat([self.upconv.latent_vector, latent_vector_conv_transpose], 0)])
            x = self.conv2([x, self.conv1.latent_vector])
        else:
            from_up = self.upconv(from_up)
            x = torch.cat((from_up, from_down), 1)
            x = self.conv1(x)
            x = self.conv2(x)
        return x


class UNet_DHP(DHP_Base):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597
    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).
    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, args):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet_DHP, self).__init__(args)
        width_mult = args.width_mult
        out_channels = args.n_colors
        in_channels = args.n_colors
        depth = 5
        up_mode='transpose'
        start_filts=int(args.n_feats * width_mult)
        merge_mode='concat'

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for upsampling. Only \"transpose\" and \"upsample\" are allowed.".format(up_mode))
    
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for merging up and down paths. Only \"concat\" and \"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible with merge_mode \"add\" at the moment because it doesn't make sense to use "
                             "nearest neighbour to reduce depth channels (by half).")

        self.num_classes = out_channels
        self.in_channels = in_channels
        self.start_filts = start_filts

        self.latent_vector = nn.Parameter(torch.randn((1)))
        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False

            down_conv = DownConv_dhp(ins, outs, pooling=pooling, args=args)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # careful! decoding only requires depth-1 blocks
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            m_flag=True
            # m_flag = True if i == (depth - 2) else False
            up_conv = UpConv_dhp(ins, outs, args=args)
            self.up_convs.append(up_conv)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.conv_final = conv_dhp(outs, self.num_classes, 1, batchnorm=False, args=args)

        self.reset_params()
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
        # masks = [torch.ones_like(latent_vectors[0], dtype=torch.bool, device='cuda')] + \
        #         [v.abs() >= min(self.pt, v.abs().topk(self.mc)[0][-1]) for v in latent_vectors[1:-1]] + \
        #         [torch.ones_like(latent_vectors[-1], dtype=torch.bool, device='cuda')]
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
        for i, layer in enumerate(self.down_convs):
            vm = [latent_vectors[2 * i]] + masks[2 * i: 2 * i + 3]
            layer.set_parameters(vm, calc_weight)
        for i, layer in enumerate(self.up_convs):
            vm = [latent_vectors[10 + 3 * i]] + masks[10 + 3 * i: 14 + 3 * i] + [latent_vectors[8 - 2 * i], masks[8 - 2 * i]]
            layer.set_parameters(vm, calc_weight)
        vm = [latent_vectors[22]] + masks[22:]
        self.conv_final.set_parameters(vm, calc_weight)

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []
        if not self.finetuning:
            latent_vectors = self.gather_latent_vector()
            # embed()
            for i, module in enumerate(self.down_convs):
                x, before_pool = module([x, latent_vectors[2 * i]])
                encoder_outs.append(before_pool)
            for i, module in enumerate(self.up_convs):
                before_pool = encoder_outs[-(i + 2)]
                x = module(before_pool, [x, latent_vectors[10 + 3 * i], latent_vectors[8 - 2 * i]])
            x = self.conv_final([x, latent_vectors[22]])
        else:
            # encoder pathway, save outputs for merging
            for i, module in enumerate(self.down_convs):
                x, before_pool = module(x)
                encoder_outs.append(before_pool)

            for i, module in enumerate(self.up_convs):
                # embed()
                before_pool = encoder_outs[-(i + 2)]
                x = module(before_pool, x)
            # No softmax is used. This means you need to use
            # nn.CrossEntropyLoss is your training script,
            # as this module includes a softmax already.
            # mem = torch.cuda.max_memory_allocated()/1024.0**3
            # print('Memory used inside model {}'.format(mem))
            x = self.conv_final(x)

        return x


# for i, (k, v) in enumerate(zip(kk, latent_vectors)):
#     print(i, list(v.shape), k)


