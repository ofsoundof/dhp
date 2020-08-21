'''
This module applies diffrentiable pruning via hypernetworks to MobileNetV2.

MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_dhp.dhp_base import DHP_Base, conv_dhp
from IPython import embed

def make_model(args, parent=False):
    return MobileNetV2_DHP(args)


def convert_index(i):
    if i == 0:
        x = 0
    elif i == 1:
        x = 8
    elif i == 2:
        x = 1
    elif 3 <= i < 5:
        x = 2
    elif 5 <= i < 8:
        x = 3
    elif 8 <= i < 12:
        x = 4
    elif 12 <= i < 15:
        x = 5
    elif 15 <= i < 18:
        x = 6
    else:
        x = 7
    return x


class InvertedResidual_dhp(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, latent_vector=None, embedding_dim=8):
        super(InvertedResidual_dhp, self).__init__()
        self.in_channels = inp
        self.out_channels = oup
        self.stride = stride
        self.finetuning = False
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        self.layers = nn.ModuleList()
        if expand_ratio != 1:
            # pw
            self.layers.append(conv_dhp(inp, hidden_dim, 1, embedding_dim=embedding_dim))
        self.layers.append(conv_dhp(hidden_dim, hidden_dim, 3, stride=stride, groups=hidden_dim, embedding_dim=embedding_dim))
        self.layers.append(conv_dhp(hidden_dim, oup, 1, act=False, latent_vector=latent_vector, embedding_dim=embedding_dim))

    def set_parameters(self, vector_mask, calc_weight=False):
        self.layers[0].set_parameters(vector_mask[:3], calc_weight)
        if len(self.layers) == 2:
            self.layers[1].set_parameters(vector_mask[:2] + [vector_mask[3]], calc_weight)
        else:
            self.layers[1].set_parameters([self.layers[0].latent_vector] + vector_mask[2:4], calc_weight)
            if self.in_channels == self.out_channels:
                m = [vector_mask[1]]
            else:
                m = [vector_mask[-1]]
            self.layers[2].set_parameters([self.layers[0].latent_vector] + [vector_mask[2]] + m, calc_weight)

    def forward(self, x):

        if not self.finetuning:
            x, latent_vector_input = x
            out = self.layers[0]([x, latent_vector_input])
            if len(self.layers) == 2:
                out = self.layers[1]([out, latent_vector_input])
            else:
                out = self.layers[1]([out, self.layers[0].latent_vector])
                out = self.layers[2]([out, self.layers[0].latent_vector])
        else:
            out = self.layers[0](x)
            for layer in self.layers[1:]:
                out = layer(out)

        if self.use_res_connect:
            out += x
        return out


# (expansion, out_planes, num_blocks, stride)
cfg = [(1,  16, 1, 1),
       (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
       (6,  32, 3, 2),
       (6,  64, 4, 2),
       (6,  96, 3, 1),
       (6, 160, 3, 2),
       (6, 320, 1, 1)]

cfg_imagenet = [(1,  16, 1, 1),
       (6,  24, 2, 2),
       (6,  32, 3, 2),
       (6,  64, 4, 2),
       (6,  96, 3, 1),
       (6, 160, 3, 2),
       (6, 320, 1, 1)]


class MobileNetV2_DHP(DHP_Base):

    def __init__(self, args):
        super(MobileNetV2_DHP, self).__init__(args=args)
        self.width_mult = args.width_mult
        self.prune_classifier = args.prune_classifier

        if args.data_train == 'ImageNet':
            self.cfg = cfg_imagenet
        else:
            self.cfg = cfg

        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        stride = 1 if args.data_train.find('CIFAR') >= 0 else 2

        self.latent_vectors = nn.ParameterList([nn.Parameter(torch.randn((3)))] +
                                               [nn.Parameter(torch.randn((int(c[1] * self.width_mult)))) for c in self.cfg])

        self.features = nn.ModuleList([conv_dhp(3, int(32 * self.width_mult), kernel_size=3, stride=stride, embedding_dim=self.embedding_dim)])
        self.features.extend(self._make_layers(in_planes=int(32 * self.width_mult)))
        self.features.append(conv_dhp(int(320 * self.width_mult), int(1280 * self.width_mult), kernel_size=1, stride=1, embedding_dim=self.embedding_dim))
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(int(1280 * self.width_mult), self.n_classes))
        self.show_latent_vector()

    def _make_layers(self, in_planes):
        layers = []
        for i, (expansion, out_planes, num_blocks, stride) in enumerate(self.cfg):
            out_planes = int(out_planes * self.width_mult)
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(InvertedResidual_dhp(in_planes, out_planes, stride, expansion, latent_vector=self.latent_vectors[i + 1], embedding_dim=self.embedding_dim))
                in_planes = out_planes
        return layers

    def mask(self):
        masks = []
        for i, v in enumerate(self.gather_latent_vector(grad_prune=self.grad_prune, grad_normalize=self.grad_normalize)):
            # if (not self.prune_classifier and i != 0 and i != 42 and not (i > 8 and i % 2 == 1)) \
            #         and (self.prune_classifier and i != 0 and not (i > 8 and i % 2 == 1)):
                # embed()
            if not self.prune_classifier:
                flag = i != 0 and i != 42 and not (i > 8 and i % 2 == 1)
            else:
                flag = i != 0 and not (i > 8 and i % 2 == 1)
            if flag:
                if flag == 42:
                    channels = self.remain_channels(v, percentage=self.linear_percentage)
                else:
                    channels = self.remain_channels(v)
                masks.append(v.abs() >= min(self.pt, v.abs().topk(channels)[0][-1]))
            else:
                masks.append(torch.ones_like(v, dtype=torch.bool, device='cuda'))
        # mask has no gradients. Comparison operations are without gradients automatically.
        # when calculating the weights, biases, don't need to slice them?
        return masks

    def proximal_operator(self, lr):
        regularization = self.regularization * lr
        for i, v in enumerate(self.gather_latent_vector()):
            if not self.prune_classifier:
                flag = i != 0 and i != 42 and not (i > 8 and i % 2 == 1)
            else:
                flag = i != 0 and not (i > 8 and i % 2 == 1)
            if flag:
                if flag == 42:
                    channels = self.remain_channels(v, percentage=self.linear_percentage)
                else:
                    channels = self.remain_channels(v)
                # if (not self.prune_classifier and i != 0 and i != 42 and not (i > 8 and i % 2 == 1):
                if torch.sum(v.abs() >= self.pt) > channels:
                    self.soft_thresholding(v, regularization)

    def set_parameters(self, calc_weight=False):
        latent_vectors, masks = self.gather_latent_vector(), self.mask()

        # former_vectors, last_vectors, other_vectors, former_masks, last_masks, other_masks = self.mask()
        for i, layer in enumerate(self.features):
            j = convert_index(i)
            if i == 0:
                vm = [latent_vectors[j], masks[j], masks[8]]
            elif i == 1:
                vm = [latent_vectors[j], masks[j], masks[9], masks[1]]
            elif i == 18:
                vm = [latent_vectors[j], masks[j], masks[-1]]
            elif i == 2 or i == 4 or i == 7 or i == 11 or i == 14 or i == 17:
                vm = [latent_vectors[j], masks[j]] + masks[2 * i + 6: 2 * i + 8] + [masks[j + 1]]
            else:
                vm = [latent_vectors[j], masks[j]] + masks[2 * i + 6: 2 * i + 8]
            layer.set_parameters(vm, calc_weight)
        if self.prune_classifier:
            super(MobileNetV2_DHP, self).set_parameters(calc_weight)
            mask = self.mask()[42]
            mask_input = mask.to(torch.float32).nonzero().squeeze(1)
            self.classifier[1].in_features_remain = mask_input.shape[0]
            if calc_weight:
                self.classifier[1].in_features = mask_input.shape[0]
                # embed()
                # if mask_input.shape[0] < 1280:
                #     mask_input = mask_input.cpu().numpy().tolist()
                #     num = len(mask_input)
                #     mask_other = sorted(list(set(range(mask.shape[0])) - set(mask_input)))
                #     index = sorted(random.sample(range(len(mask_other)), 1280 - num))
                #     for i in index:
                #         mask_input.append(mask_other[i])
                #     mask_input = torch.tensor(sorted(mask_input)).cuda()
                weight = self.classifier[1].weight.data
                weight = torch.index_select(weight, dim=1, index=mask_input)
                self.classifier[1].weight = nn.Parameter(weight)

    def forward(self, x):
        if not self.finetuning:
            latent_vectors = self.gather_latent_vector()
            # for i, (k, v) in enumerate(zip(kk, latent_vectors)):
            #     print(i, list(v.shape), k)
            # embed()
            for i, layer in enumerate(self.features):
                j = convert_index(i)
                x = layer([x, latent_vectors[j]])
        else:
            for i, layer in enumerate(self.features):
                x = layer(x)
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        # out = F.avg_pool2d(x, 4)
        # out = out.view(out.size(0), -1)
        out = x.mean([2, 3])
        out = self.classifier(out)
        return out


# class conv_dhp(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=False, groups=1, batchnorm=True, act=True, latent_vector=None, args=None):
#         """
#         :param in_channels:
#         :param out_channels:
#         :param kernel_size:
#         :param stride:
#         :param bias:
#         :param batchnorm: whether to append Batchnorm after the activations.
#         :param act: whether to append ReLU after the activations.
#         :param args:
#         """
#         super(conv_dhp, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.groups = groups
#         self.batchnorm = batchnorm
#         self.act = act
#         self.finetuning = False
#         self.in_channels_per_group = self.in_channels // self.groups
#         # hypernetwork
#         latent_dim = args.latent_dim
#         bound = math.sqrt(3) * math.sqrt(1 / 1)
#         weight1 = torch.randn((self.in_channels_per_group, out_channels, latent_dim)).uniform_(-bound, bound)
#         # Hyperfan-in
#         bound = math.sqrt(3) * math.sqrt(1 / (latent_dim * self.in_channels_per_group * kernel_size ** 2))
#         weight2 = torch.randn((self.in_channels_per_group, out_channels, kernel_size ** 2, latent_dim)).uniform_(-bound, bound)
#
#         if latent_vector is None:
#             if self.groups == 1:
#                 self.latent_vector = nn.Parameter(torch.randn((out_channels)))
#             else:
#                 self.latent_vector = nn.Parameter(torch.randn((self.in_channels_per_group)))
#         else:
#             self.latent_vector = latent_vector
#             # self.register_buffer('latent_vector', latent_vector.clone())
#             # self.latent_vector = latent_vector.clone().cuda()
#         self.weight1 = nn.Parameter(weight1)
#         self.weight2 = nn.Parameter(weight2)
#         self.bias0 = nn.Parameter(torch.zeros(self.in_channels_per_group, out_channels))
#         self.bias1 = nn.Parameter(torch.zeros(self.in_channels_per_group, out_channels, latent_dim))
#         self.bias2 = nn.Parameter(torch.zeros(self.in_channels_per_group, out_channels, kernel_size ** 2))
#         # self.bn_hyper = nn.BatchNorm2d(in_channels)
#
#         # main network
#         self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
#         if self.batchnorm:
#             self.bn_main = nn.BatchNorm2d(out_channels)
#         if self.act:
#             self.relu = nn.ReLU6()
#
#     def __repr__(self):
#         s = '\n(1): Conv_dhp({}, {}, kernel_size=({}, {}), stride={}, groups={})'
#         if hasattr(self, 'in_channels_remain') and hasattr(self, 'out_channels_remain') and hasattr(self, 'groups_remain'):
#             s = s.format(self.in_channels_remain, self.out_channels_remain, self.kernel_size, self.kernel_size, self.stride, self.groups_remain)
#         else:
#             s = s.format(self.in_channels, self.out_channels, self.kernel_size, self.kernel_size, self.stride, self.groups)
#         if self.batchnorm:
#             s = s + '\n(2): ' + repr(self.bn_main)
#         if self.act:
#             s = s + '\n(3): ' + repr(self.relu)
#         s = addindent(s, 2)
#         return s
#
#     def set_parameters(self, vector_mask, calc_weight=False):
#
#         if self.groups == 1:
#             latent_vector_input, mask_input, mask_output = vector_mask #TODO: check the mask_input and mask_output here
#             latent_vector_output = self.latent_vector
#         else:
#             latent_vector_input = self.latent_vector
#             latent_vector_output, mask_output, mask_input = vector_mask
#
#         mask_input = mask_input.to(torch.float32).nonzero().squeeze(1)
#         mask_output = mask_output.to(torch.float32).nonzero().squeeze(1)
#         if calc_weight:
#
#             latent_vector_input = torch.index_select(latent_vector_input, dim=0, index=mask_input)
#             latent_vector_output = torch.index_select(latent_vector_output, dim=0, index=mask_output)
#             weight = self.calc_weight(latent_vector_input, latent_vector_output, mask_input, mask_output)
#
#             bias = self.bias if self.bias is None else torch.index_select(self.bias, dim=0, index=mask_output)
#             self.weight = nn.Parameter(weight)
#             self.bias = bias if bias is None else nn.Parameter(bias)
#
#             if self.batchnorm:
#                 bn_weight = torch.index_select(self.bn_main.weight, dim=0, index=mask_output)
#                 bn_bias = torch.index_select(self.bn_main.bias, dim=0, index=mask_output)
#                 bn_mean = torch.index_select(self.bn_main.running_mean, dim=0, index=mask_output)
#                 bn_var = torch.index_select(self.bn_main.running_var, dim=0, index=mask_output)
#
#                 self.bn_main.weight = nn.Parameter(bn_weight)
#                 self.bn_main.bias = nn.Parameter(bn_bias)
#                 self.bn_main.running_mean = bn_mean
#                 self.bn_main.running_var = bn_var
#         if self.groups == 1:
#             self.in_channels_remain = mask_input.shape[0]
#             self.groups_remain = self.groups
#         else:
#             self.in_channels_remain = mask_output.shape[0]
#             self.groups_remain = mask_output.shape[0]
#             if calc_weight:
#                 self.groups = mask_output.shape[0]
#         self.out_channels_remain = mask_output.shape[0]
#         if self.batchnorm:
#             self.bn_main.num_features = mask_output.shape[0]
#             self.bn_main.num_features_remain = mask_output.shape[0]
#         if self.act:
#             self.relu.channels = mask_output.shape[0]
#
#     def calc_weight(self, latent_vector_input=None, latent_vector_output=None, mask_input=None, mask_output=None):
#
#         if latent_vector_output is None:
#             latent_vector_output = self.latent_vector
#         elif latent_vector_input is None:
#             latent_vector_input = self.latent_vector
#         elif latent_vector_output is None and latent_vector_input is None:
#             raise NotImplementedError('During the pruning of MobileNetV2 convs, please provide at least one of the latent input and output vectors')
#         if mask_input is None and mask_output is None:
#             bias0, bias1, bias2, weight1, weight2 = self.bias0, self.bias1, self.bias2, self.weight1, self.weight2
#         else:
#             bias0 = torch.index_select(torch.index_select(self.bias0, dim=0, index=mask_input), dim=1,
#                                        index=mask_output)
#             bias1 = torch.index_select(torch.index_select(self.bias1, dim=0, index=mask_input), dim=1,
#                                        index=mask_output)
#             bias2 = torch.index_select(torch.index_select(self.bias2, dim=0, index=mask_input), dim=1,
#                                        index=mask_output)
#             weight1 = torch.index_select(torch.index_select(self.weight1, dim=0, index=mask_input), dim=1,
#                                          index=mask_output)
#             weight2 = torch.index_select(torch.index_select(self.weight2, dim=0, index=mask_input), dim=1,
#                                          index=mask_output)
#
#         weight = torch.matmul(latent_vector_input.unsqueeze(-1), latent_vector_output.unsqueeze(0)) + bias0
#         weight = weight.unsqueeze(-1) * weight1 + bias1
#         weight = torch.matmul(weight2, weight.unsqueeze(-1)).squeeze(-1) + bias2
#
#         in_channels = latent_vector_input.nelement()
#         out_channels = latent_vector_output.nelement()
#         weight = weight.reshape(in_channels, out_channels, self.kernel_size, self.kernel_size).permute(1, 0, 2, 3)
#         # weight = self.bn_hyper(weight)
#         return weight
#
#     def forward(self, input):
#         if not self.finetuning:
#             out = self.forward_pruning(input)
#         else:
#             out = self.forward_finetuning(input)
#         if self.batchnorm:
#             out = self.bn_main(out)
#         if self.act:
#             out = self.relu(out)
#         return out
#
#     def forward_pruning(self, x):
#         # execute the hypernetworks to get the weights of the backbone network
#         if self.groups == 1:
#             x, latent_vector_input = x
#             weight = self.calc_weight(latent_vector_input=latent_vector_input)
#         else:
#             x, latent_vector_output = x
#             weight = self.calc_weight(latent_vector_output=latent_vector_output)
#         # embed()
#         out = F.conv2d(x, weight, bias=self.bias, stride=self.stride, padding=self.kernel_size // 2, groups=self.groups)
#         return out
#
#     def forward_finetuning(self, input):
#         out = F.conv2d(input, self.weight, bias=self.bias, stride=self.stride, padding=self.kernel_size // 2, groups=self.groups)
#         return out


#
# def test():
#     net = MobileNetV2_DHP()
#     x = torch.randn(2,3,32,32)
#     y = net(x)
#     print(y.size())


# for i, (k, v) in enumerate(zip(kk, latent_vectors)):
#     print(i, list(v.shape), k)