"""
This module is used to develop differentiable pruning via hypernetworks on DnCNN for image denoising task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import numpy as np
import os
# from IPython import embed


class conv_dhp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True, groups=1, batchnorm=True, act=None,
                 latent_vector=None, args=None, scale=1, transpose=False):
        """
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param bias:
        :param batchnorm: whether to append Batchnorm after the activations.
        :param act: whether to append ReLU after the activations.
        :param args:
        """
        super(conv_dhp, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias_flag = bias
        self.groups = 1
        self.batchnorm = batchnorm
        self.act = True if act is not None else False
        self.finetuning = False
        self.transpose = transpose
        self.scale = scale


        # hypernetwork
        embedding_dim = args.embedding_dim
        bound = math.sqrt(3) * math.sqrt(1 / 1)
        weight1 = torch.randn((in_channels, out_channels, embedding_dim)).uniform_(-bound, bound)
        # Hyperfan-in
        bound = math.sqrt(3) * math.sqrt(1 / (embedding_dim * in_channels * kernel_size ** 2))
        weight2 = torch.randn((in_channels, out_channels, kernel_size ** 2, embedding_dim)).uniform_(-bound, bound)

        if scale == 1:
            if latent_vector is None:
                self.latent_vector = nn.Parameter(torch.randn((out_channels)))
            else:
                self.latent_vector = latent_vector
        self.weight1 = nn.Parameter(weight1)
        self.weight2 = nn.Parameter(weight2)
        self.bias0 = nn.Parameter(torch.zeros(in_channels, out_channels))
        self.bias1 = nn.Parameter(torch.zeros(in_channels, out_channels, embedding_dim))
        self.bias2 = nn.Parameter(torch.zeros(in_channels, out_channels, kernel_size ** 2))
        # self.bn_hyper = nn.BatchNorm2d(in_channels)

        # main network
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        if self.batchnorm:
            self.bn_main = nn.BatchNorm2d(out_channels)
        if self.act:
            self.relu = act

    def __repr__(self):
        if hasattr(self, 'in_channels_remain') and hasattr(self, 'out_channels_remain'):
            s = '\n(1): Conv_dhp({}, {}, kernel_size=({}, {}), stride={}, bias={}, groups={}, scale={}, transpose={})'.\
                format(self.in_channels_remain, self.out_channels_remain, self.kernel_size, self.kernel_size,
                       self.stride, self.bias_flag, self.groups, self.scale, self.transpose)
        else:
            s = '\n(1): Conv_dhp({}, {}, kernel_size=({}, {}), stride={}, bias={}, groups={}, scale={}, transpose={})'.\
                format(self.in_channels, self.out_channels, self.kernel_size, self.kernel_size,
                       self.stride, self.bias_flag, self.groups, self.scale, self.transpose)
        if self.batchnorm:
            s = s + '\n(2): ' + repr(self.bn_main)
        if self.act:
            s = s + '\n(3): ' + repr(self.relu)
        s = addindent(s, 2)
        return s

    def set_parameters(self, vector_mask, calc_weight=False):
        if self.scale == 1:
            latent_vector_input, mask_input, mask_output = vector_mask
            latent_vector_output = self.latent_vector
        else:
            latent_vector_input, mask_input, latent_vector_output, mask_output = vector_mask
        mask_input = mask_input.to(torch.float32).nonzero().squeeze(1)
        mask_output = mask_output.to(torch.float32).nonzero().squeeze(1)
        if calc_weight:
            latent_vector_input = torch.index_select(latent_vector_input, dim=0, index=mask_input)
            latent_vector_output = torch.index_select(latent_vector_output, dim=0, index=mask_output)
            weight = self.calc_weight(latent_vector_input, latent_vector_output, mask_input, mask_output)

            bias = self.bias if self.bias is None else torch.index_select(self.bias, dim=0, index=mask_output)
            self.weight = nn.Parameter(weight)
            self.bias = bias if bias is None else nn.Parameter(bias)

            if self.batchnorm:
                bn_weight = torch.index_select(self.bn_main.weight, dim=0, index=mask_output)
                bn_bias = torch.index_select(self.bn_main.bias, dim=0, index=mask_output)
                bn_mean = torch.index_select(self.bn_main.running_mean, dim=0, index=mask_output)
                bn_var = torch.index_select(self.bn_main.running_var, dim=0, index=mask_output)

                self.bn_main.weight = nn.Parameter(bn_weight)
                self.bn_main.bias = nn.Parameter(bn_bias)
                self.bn_main.running_mean = bn_mean
                self.bn_main.running_var = bn_var
        self.in_channels_remain = mask_input.shape[0]
        self.out_channels_remain = mask_output.shape[0]

        if self.batchnorm:
            self.bn_main.num_features = mask_output.shape[0]
            self.bn_main.num_features_remain = mask_output.shape[0]
        if self.act:
            self.relu.channels = mask_output.shape[0]

    def calc_weight(self, latent_vector_input, latent_vector_output=None, mask_input=None, mask_output=None):
        if latent_vector_output is None:
            latent_vector_output = self.latent_vector
        if mask_input is None and mask_output is None:
            bias0, bias1, bias2, weight1, weight2 = self.bias0, self.bias1, self.bias2, self.weight1, self.weight2
        else:
            bias0 = torch.index_select(torch.index_select(self.bias0, dim=0, index=mask_input), dim=1,
                                       index=mask_output)
            bias1 = torch.index_select(torch.index_select(self.bias1, dim=0, index=mask_input), dim=1,
                                       index=mask_output)
            bias2 = torch.index_select(torch.index_select(self.bias2, dim=0, index=mask_input), dim=1,
                                       index=mask_output)
            weight1 = torch.index_select(torch.index_select(self.weight1, dim=0, index=mask_input), dim=1,
                                         index=mask_output)
            weight2 = torch.index_select(torch.index_select(self.weight2, dim=0, index=mask_input), dim=1,
                                         index=mask_output)
        # embed()
        weight = torch.matmul(latent_vector_input.unsqueeze(-1), latent_vector_output.unsqueeze(0)) + bias0
        weight = weight.unsqueeze(-1) * weight1 + bias1
        weight = torch.matmul(weight2, weight.unsqueeze(-1)).squeeze(-1) + bias2
        # if weight.nelement() != self.in_channels * self.out_channels * self.kernel_size ** 2:
        #     embed()
        in_channels = latent_vector_input.nelement()
        out_channels = latent_vector_output.nelement()
        if not self.transpose:
            weight = weight.reshape(in_channels, out_channels, self.kernel_size, self.kernel_size).permute(1, 0, 2, 3)
        else:
            weight = weight.reshape(in_channels, out_channels, self.kernel_size, self.kernel_size)
        # weight = self.bn_hyper(weight)
        return weight

    def forward(self, input):
        if not self.finetuning:
            out = self.forward_pruning(input)
        else:
            out = self.forward_finetuning(input)
        if self.batchnorm:
            out = self.bn_main(out)
        if self.act:
            out = self.relu(out)
        return out

    def forward_pruning(self, x):
        x, latent_vector_input = x
        # execute the hypernetworks to get the weights of the backbone network
        if self.scale != 1:
            latent_vector_output = latent_vector_input.repeat_interleave(self.scale ** 2)
            weight = self.calc_weight(latent_vector_input=latent_vector_input, latent_vector_output=latent_vector_output)
        else:
            weight = self.calc_weight(latent_vector_input)

        if not self.transpose:
            out = F.conv2d(x, weight, bias=self.bias, stride=self.stride, padding=self.kernel_size // 2)
        else:
            out = F.conv_transpose2d(x, weight, bias=self.bias, stride=self.stride)
        return out

    def forward_finetuning(self, input):
        if not self.transpose:
            out = F.conv2d(input, self.weight, bias=self.bias, stride=self.stride, padding=self.kernel_size // 2)
        else:
            out = F.conv_transpose2d(input, self.weight, bias=self.bias, stride=self.stride)
        return out


class DHP_Base(nn.Module):
    def __init__(self, args):
        super(DHP_Base, self).__init__()
        self.args = args
        self.finetuning = False  # used to select the forward pass
        self.prune_upsampler = args.prune_upsampler
        self.regularization = args.regularization_factor  # the regularization factor for l1 sparsity regularizer
        self.pt = args.prune_threshold  # The pruning threshold
        self.mc = 4  # The minimum number of remaining channels
        self.embedding_dim = args.embedding_dim
        self.n_colors = args.n_colors
        self.grad_prune = args.grad_prune
        self.grad_normalize = args.grad_normalize
        self.rp = args.remain_percentage

    def gather_latent_vector(self, return_key=False, grad_prune=False, grad_normalize=False):
        """
        :return: all of the latent vectors
        """
        latent_vectors = []
        s = []
        kk = []
        for k, v in self.state_dict(keep_vars=True).items():
            if k.find('latent_vector') >= 0 and not id(v) in s:
                if grad_prune and v.grad is not None:
                    vector = v.grad
                    if grad_normalize:
                        vector /= vector.max()
                else:
                    vector = v

                latent_vectors.append(vector)
                s.append(id(v))
                if return_key:
                    kk.append(k)
        if return_key:
            return latent_vectors, kk
        else:
            return latent_vectors

    def show_latent_vector(self):
        latent_vectors, kk = self.gather_latent_vector(return_key=True)
        for i, (k, v) in enumerate(zip(kk, latent_vectors)):
            print(i, list(v.shape), k)

    def remain_channels(self, vector, percentage=None):
        if percentage is not None:
            channels = int(vector.shape[0] * percentage)
        else:
            if self.rp == -1:
                channels = self.mc
            else:
                channels = max(int(vector.shape[0] * self.rp), 1)
        return channels

    # def gather_latent_vector(self):
    #     """
    #     :return: former_vectors -> the latent vectors of the former layers in the Bottleneck or ResBlock.
    #              other_vectors -> the other latent vectors in ResNet including, the 1) input latent vector
    #              of the first conv, the 2), 3) & 4) latent vectors the three stages.
    #     """
    #     latent_vectors = []
    #     s = []
    #     kk = []
    #     for k, v in self.state_dict(keep_vars=True).items():
    #         if k.find('latent_vector') >= 0 and not id(v) in s:
    #             latent_vectors.append(v)
    #             s.append(id(v))
    #             kk.append(k)
    #     return latent_vectors
    #     # return latent_vectors, kk

    def mask(self):
        pass

    def soft_thresholding(self, latent_vector, reg):
        vector = latent_vector.data
        vector = torch.max(torch.abs(vector) - reg, torch.zeros_like(vector, device=vector.device)) * torch.sign(vector)
        latent_vector.data = vector

    def proximal_operator(self, lr):
        pass

    def set_parameters(self, calc_weight=False):
        pass

    def forward(self, x):
        pass

    def delete_hyperparameters(self):
        for m in self.modules():
            if isinstance(m, conv_dhp):
                if hasattr(m, 'latent_vector'):
                    del m.latent_vector
                del m.weight1, m.weight2, m.bias0, m.bias1, m.bias2

    def reset_after_searching(self):
        self.apply(set_finetune_flag)
        self.set_parameters(calc_weight=True)
        self.delete_hyperparameters()

    # def set_channels(self):
    #     for m in self.modules():
    #         if isinstance(m, conv_dhp):
    #             s = m.weight.shape
    #             m.in_channels, m.out_channels = s[1], s[0]
    #         elif isinstance(m, nn.BatchNorm2d):
    #             s = m.weight.shape
    #             m.num_features = s[0]
    #         elif isinstance(m, nn.Linear):
    #             s = m.weight.shape
    #             m.in_features, m.out_features = s[1], s[0]

    def set_channels(self):
        for m in self.modules():
            if isinstance(m, conv_dhp):
                s = m.weight.shape
                m.in_channels_remain, m.in_channels = s[1], s[1]
                m.out_channels_remain, m.out_channels = s[0], s[0]
            elif isinstance(m, nn.BatchNorm2d):
                s = m.weight.shape
                m.num_features, m.num_features_remain = s[0], s[0]
            elif isinstance(m, nn.Linear):
                s = m.weight.shape
                m.in_features_remain, m.in_features = s[1], s[1]
                m.out_features_remain, m.out_features = s[0], s[0]
            elif isinstance(m, (nn.ReLU, nn.PReLU)):
                if hasattr(m, 'channels'):
                    del m.channels

    def load_state_dict(self, state_dict, strict=True):
        if strict:
            # used to load the model parameters during training
            super(DHP_Base, self).load_state_dict(state_dict, strict)
        else:
            # used to load the model parameters during test
            own_state = self.state_dict(keep_vars=True)
            for name, param in state_dict.items():
                if name in own_state:
                    if isinstance(param, nn.Parameter):
                        param = param.data
                    if param.size() != own_state[name].size():
                        own_state[name].data = param
                    else:
                        own_state[name].data.copy_(param)
            self.set_channels()

    def latent_vector_distribution(self, epoch, batch, fpath):
        filename = os.path.join(fpath, 'features/latent{}/epoch{}_batch{}.png')
        for i, v in enumerate(self.gather_latent_vector()):
            if not os.path.exists(os.path.join(fpath, 'features/latent{}'.format(i + 1))):
                os.makedirs(os.path.join(fpath, 'features/latent{}'.format(i + 1)))
            plot_figure_dhp(v.data, i + 1, filename.format(i + 1, epoch, batch))

    def per_layer_compression_ratio(self, epoch, batch, fpath, save_pt=False):
        layers = [m for m in self.modules() if isinstance(m, conv_dhp)]
        per_layer_ratio = []
        for l, layer in enumerate(layers):
            per_layer_ratio.append(layer.out_channels_remain / layer.out_channels)
        if not save_pt:
            plot_per_layer_compression_ratio(per_layer_ratio,
                                             os.path.join(fpath, 'per_layer_compression_ratio/epoch{}_batch{}.png'
                                                          .format(epoch, batch)))
        else:
            torch.save(per_layer_ratio, os.path.join(fpath, 'per_layer_compression_ratio_final.pt'))


def plot_figure_dhp(latent_vector, l, filename):
    axis = np.array(list(range(1, latent_vector.shape[0]+1)))
    latent_vector = latent_vector.abs().detach().cpu().numpy()
    fig = plt.figure()
    plt.title('Latent vector {}, Max: {:.4f}, Ave: {:.4f}, Min: {:.4f}'.
              format(l, latent_vector.max(), latent_vector.mean(), latent_vector.min()))
    plt.plot(axis, latent_vector, label='Unsorted')
    plt.plot(axis, np.sort(latent_vector), label='Sorted')
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Latent Vector')
    plt.grid(True)
    plt.savefig(filename, dpi=50)
    # plt.show()
    plt.close(fig)


def plot_per_layer_compression_ratio(per_layer_ratio, filename):
    axis = np.array(list(range(1, len(per_layer_ratio) + 1)))
    per_layer_ratio = np.array(per_layer_ratio)
    fig = plt.figure()
    plt.title('Per Layer Compression Ratio')
    plt.plot(axis, per_layer_ratio)
    plt.xlabel('Index')
    plt.ylabel('Compression Ratio')
    plt.grid(True)
    plt.savefig(filename, dpi=300)
    # plt.show()
    plt.close(fig)


def plot_compression_ratio(compression_ratio, filename, frequency_per_epoch=1):
    if frequency_per_epoch == 1:
        axis = np.array(list(range(1, len(compression_ratio) + 1)))
    else:
        axis = np.array(list(range(1, len(compression_ratio) + 1)), dtype=float) / frequency_per_epoch
    compression_ratio = np.array(compression_ratio)
    fig = plt.figure()
    plt.title('Network Compression Ratio')
    plt.plot(axis, compression_ratio)
    # plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Ratio')
    plt.grid(True)
    plt.savefig(filename, dpi=300)
    plt.close(fig)


def addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


def set_finetune_flag(module):
    module.finetuning = True