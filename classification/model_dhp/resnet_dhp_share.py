"""
Designed three different forward passes.
1. thorough slicing: 1) slicing the weights and biaes in the hypernetwork. 2) for Batchnorm, use nn.ModuleList to
                        include batchnorms with different features.
           Problems: 1) slicing during every iteration is time-consuming. 2) Batchnorm problem -> need to
                        train every batchnorm in the ModuleList. The channels are not perfectly matched.
             Status: not implemented
2. no slicing and no masking: This is the easiest implementation. During the optimization/searching stage, do not slice
                        the weights or biases of the hypernetworks. Masks are not applied either. Corresponding to
                        module 'resnet_dhp.py' and 'resnet_dhp_share.py'. Note that for 'resnet_dhp_share.py', the
                        latent vectors are shared by ResBlocks in the same stage.
             Status: developing and chosen
3. no slicing but with masking: Do not slice the weights or biases of the hypernetworks but masks are applied to them.
                        This forward pass corresponds to module 'resnet_dhp_mask.py'.
             Status: developed but decided not to use it.
"""
import torch
import torch.nn as nn
from model_dhp.dhp_base import DHP_Base, conv_dhp
from IPython import embed


def make_model(args, parent=False):
    return ResNet_DHP_Share(args)


class ResBlock_dhp(nn.Module):
    def __init__(self, in_channels, planes, kernel_size, stride=1, conv3x3=conv_dhp, downsample=None, latent_vector=None, embedding_dim=8):
        super(ResBlock_dhp, self).__init__()
        expansion = 1
        self.finetuning = False
        self.stride = stride
        self.layer1 = conv3x3(in_channels, planes, kernel_size, stride=stride, embedding_dim=embedding_dim)
        self.layer2 = conv3x3(planes, expansion * planes, kernel_size, act=False, latent_vector=latent_vector, embedding_dim=embedding_dim)
        self.downsample = downsample

        # if stride != 1 or in_channels != planes:
        #     self.downsample = conv3x3(in_channels, planes * expansion, 1, stride=stride, act=False, args=args)
        self.act_out = nn.ReLU()

    def set_parameters(self, vector_mask, calc_weight=False):

        self.layer1.set_parameters(vector_mask[:3], calc_weight)
        if self.downsample is not None:
            self.layer2.set_parameters([self.layer1.latent_vector] + vector_mask[2:], calc_weight)
        else:
            self.layer2.set_parameters([self.layer1.latent_vector] + vector_mask[2:0:-1], calc_weight)
        if self.downsample is not None:
            self.downsample.set_parameters(vector_mask[:2] + [vector_mask[-1]], calc_weight)
        self.act_out.channels = self.layer2.out_channels_remain if hasattr(self.layer2, 'out_channels_remain') \
            else self.layer2.out_channels

    def forward(self, x):
        if not self.finetuning:
            x, latent_input_vector = x
            # Need the latent vector of the previous layer.
            out = self.layer1([x, latent_input_vector])
            out = self.layer2([out, self.layer1.latent_vector])
            if self.downsample is not None:
                x = self.downsample([x, latent_input_vector])
        else:
            out = self.layer2(self.layer1(x))
            if self.downsample is not None:
                x = self.downsample(x)
        out += x
        out = self.act_out(out)
        return out


class BottleNeck_dhp(nn.Module):
    def __init__(self, in_channels, planes, kernel_size, stride=1, conv3x3=conv_dhp, downsample=None, latent_vector=None, embedding_dim=8):
        super(BottleNeck_dhp, self).__init__()
        expansion = 4
        self.finetuning = False
        self.stride = stride
        self.layer1 = conv3x3(in_channels, planes, 1, embedding_dim=embedding_dim)
        self.layer2 = conv3x3(planes, planes, kernel_size, stride=stride, embedding_dim=embedding_dim)
        self.layer3 = conv3x3(planes, expansion*planes, 1, act=False, latent_vector=latent_vector, embedding_dim=embedding_dim)
        self.downsample = downsample

        # if stride != 1 or in_channels != planes:
        #     self.downsample = conv3x3(in_channels, planes * expansion, 1, stride=stride, act=False, args=args)
        self.act_out = nn.ReLU()

    def set_parameters(self, vector_mask, calc_weight=False):
        self.layer1.set_parameters(vector_mask[:3], calc_weight)
        self.layer2.set_parameters([self.layer1.latent_vector] + vector_mask[2:4], calc_weight)
        if self.downsample is not None:
            self.layer3.set_parameters([self.layer2.latent_vector] + vector_mask[3:], calc_weight)
        else:
            self.layer3.set_parameters([self.layer2.latent_vector] + vector_mask[3:0:-2], calc_weight) #TODO: make sure this is correct
        if self.downsample is not None:
            self.downsample.set_parameters(vector_mask[:2] + [vector_mask[-1]], calc_weight)
        self.act_out.channels = self.layer3.out_channels_remain if hasattr(self.layer3, 'out_channels_remain') \
            else self.layer3.out_channels

    def forward(self, x):
        if not self.finetuning:
            x, latent_input_vector = x
            # Need the latent vector of the previous layer.
            out = self.layer1([x, latent_input_vector])
            out = self.layer2([out, self.layer1.latent_vector])
            out = self.layer3([out, self.layer2.latent_vector])
            if self.downsample is not None:
                x = self.downsample([x, latent_input_vector])
        else:
            out = self.layer3(self.layer2(self.layer1(x)))
            if self.downsample is not None:
                x = self.downsample(x)
        out += x
        out = self.act_out(out)

        return out


class ResNet_DHP_Share(DHP_Base):
    def __init__(self, args):
        super(ResNet_DHP_Share, self).__init__(args=args)
        self.width_mult = args.width_mult

        if args.depth <= 56:
            self.expansion = 1
            self.block = ResBlock_dhp
            self.n_blocks = (args.depth - 2) // 6
        else:
            self.expansion = 4
            self.block = BottleNeck_dhp
            self.n_blocks = (args.depth - 2) // 9
        self.in_channels = int(16 * self.width_mult)
        self.downsample_type = 'C'

        self.latent_vector_stage0 = nn.Parameter(torch.randn((3)))
        self.latent_vector_stage1 = nn.Parameter(torch.randn((int(16 * self.width_mult) * self.expansion)))
        self.latent_vector_stage2 = nn.Parameter(torch.randn((int(32 * self.width_mult) * self.expansion)))
        self.latent_vector_stage3 = nn.Parameter(torch.randn((int(64 * self.width_mult) * self.expansion)))

        stride = 1 if args.data_train.find('CIFAR') >= 0 else 2
        v = self.latent_vector_stage1 if self.expansion == 1 else None
        self.features = nn.ModuleList([conv_dhp(args.n_colors, int(16 * self.width_mult), kernel_size=self.kernel_size, stride=stride, latent_vector=v, embedding_dim=self.embedding_dim)])
        self.features.extend(self.make_layer(self.n_blocks, int(16 * self.width_mult), self.kernel_size, latent_vector=self.latent_vector_stage1))
        self.features.extend(self.make_layer(self.n_blocks, int(32 * self.width_mult), self.kernel_size, stride=2, latent_vector=self.latent_vector_stage2))
        self.features.extend(self.make_layer(self.n_blocks, int(64 * self.width_mult), self.kernel_size, stride=2, latent_vector=self.latent_vector_stage3))
        self.pooling = nn.AvgPool2d(8)
        self.classifier = nn.Linear(int(64 * self.width_mult) * self.expansion, self.n_classes)
        self.show_latent_vector()

    def make_layer(self, blocks, planes, kernel_size, stride=1, conv3x3=conv_dhp, latent_vector=None):
        out_channels = planes * self.expansion
        if stride != 1 or self.in_channels != out_channels:
            downsample = conv3x3(self.in_channels, out_channels, 1, stride=stride, act=False, latent_vector=latent_vector, embedding_dim=self.embedding_dim)
        else:
            downsample = None
        kwargs = {'conv3x3': conv3x3, 'embedding_dim': self.embedding_dim}
        m = [self.block(self.in_channels, planes, kernel_size, stride=stride, downsample=downsample, latent_vector=latent_vector, **kwargs)]
        self.in_channels = out_channels
        for _ in range(blocks - 1):
            m.append(self.block(self.in_channels, planes, kernel_size, latent_vector=latent_vector, **kwargs))

        return m

    def mask(self):
        masks = []
        for i, v in enumerate(self.gather_latent_vector(grad_prune=self.grad_prune, grad_normalize=self.grad_normalize)):
            channels = self.remain_channels(v)
            if i == 0 or i == 3:
                masks.append(torch.ones_like(v, dtype=torch.bool, device='cuda'))
            else:
                masks.append(v.abs() >= min(self.pt, v.abs().topk(channels)[0][-1]))
        # mask has no gradients. Comparison operations are without gradients automatically.
        # when calculating the weights, biases, don't need to slice them?
        return masks

    def proximal_operator(self, lr):
        regularization = self.regularization * lr
        for i, v in enumerate(self.gather_latent_vector()):
            channels = self.remain_channels(v)
            if i != 0 and i != 3:
                if torch.sum(v.abs() >= self.pt) > channels:
                    if self.args.sparsity_regularizer == 'l1':
                        self.soft_thresholding(v, regularization)
                    elif self.args.sparsity_regularizer == 'l2':
                        self.proximal_euclidean_norm(v, regularization)
                    else:
                        raise NotImplementedError('Solution to regularization type {} is not implemented.'
                                                  .format(self.args.sparsity_regularizer))

    def set_parameters(self, calc_weight=False):
        latent_vectors, masks = self.gather_latent_vector(), self.mask()
        # former_vectors, last_vectors, other_vectors, former_masks, last_masks, other_masks = self.mask()
        offset = 1 if self.expansion == 1 else 2
        for i, layer in enumerate(self.features):
            if self.expansion == 1:
                if i == 0:
                    vm = [latent_vectors[0]] + masks[:2]
                else:
                    mask = masks[(i + 1) * offset: (i + 2) * offset] if self.expansion == 4 else [masks[i + 3]]
                    stage = (i - 1) // self.n_blocks # stage - 1
                    if (i - 1) % self.n_blocks == 0 and i > 1:
                        vm = [latent_vectors[stage], masks[stage]] + mask + [masks[stage + 1]]
                    else:
                        vm = [latent_vectors[stage + 1], masks[stage + 1]] + mask
            elif self.expansion == 4:
                if i == 0:
                    vm = [latent_vectors[0], masks[0], masks[4]]
                else:
                    mask = masks[i * 2 + 3: i * 2 + 5]
                    stage = (i - 1) // self.n_blocks
                    if i == 1:
                        vm = [latent_vectors[4], masks[4]] + mask + [masks[1]]
                    elif (i - 1) % self.n_blocks == 0 and i > 1:
                        vm = [latent_vectors[stage], masks[stage]] + mask + [masks[stage + 1]]
                    else:
                        vm = [latent_vectors[stage + 1], masks[stage + 1]] + mask
            else:
                raise NotImplementedError('Expansion type {} not implemented for ResNet'.format(self.expansion))
            layer.set_parameters(vm, calc_weight)

    def forward(self, x):
        if not self.finetuning:
            latent_vectors = self.gather_latent_vector()
            for i, layer in enumerate(self.features):
                if i == 0:
                    x = layer([x, latent_vectors[0]])
                elif i == 1:
                    if self.expansion == 1:
                        x = layer([x, latent_vectors[1]])
                    else:
                        x = layer([x, latent_vectors[4]])
                elif 1 < i <= self.n_blocks + 1:
                    x = layer([x, latent_vectors[1]])
                elif self.n_blocks + 1 < i <= 2 * self.n_blocks + 1:
                    x = layer([x, latent_vectors[2]])
                else:
                    x = layer([x, latent_vectors[3]])
        else:
            for i, layer in enumerate(self.features):
                x = layer(x)
        x = self.pooling(x)
        x = self.classifier(x.squeeze())
        return x



# class conv_dhp(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=False, batchnorm=True, act=True, latent_vector=None, args=None):
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
#         self.batchnorm = batchnorm
#         self.act = act
#         self.finetuning = False
#
#         # hypernetwork
#         latent_dim = args.latent_dim
#         bound = math.sqrt(3) * math.sqrt(1 / 1)
#         weight1 = torch.randn((in_channels, out_channels, latent_dim)).uniform_(-bound, bound)
#         # Hyperfan-in
#         bound = math.sqrt(3) * math.sqrt(1 / (latent_dim * in_channels * kernel_size ** 2))
#         weight2 = torch.randn((in_channels, out_channels, kernel_size ** 2, latent_dim)).uniform_(-bound, bound)
#
#         if latent_vector is None:
#             self.latent_vector = nn.Parameter(torch.randn((out_channels)))
#         else:
#             self.latent_vector = latent_vector
#             # self.register_buffer('latent_vector', latent_vector.clone())
#             # self.latent_vector = latent_vector.clone().cuda()
#         self.weight1 = nn.Parameter(weight1)
#         self.weight2 = nn.Parameter(weight2)
#         self.bias0 = nn.Parameter(torch.zeros(in_channels, out_channels))
#         self.bias1 = nn.Parameter(torch.zeros(in_channels, out_channels, latent_dim))
#         self.bias2 = nn.Parameter(torch.zeros(in_channels, out_channels, kernel_size ** 2))
#         # self.bn_hyper = nn.BatchNorm2d(in_channels)
#
#         # main network
#         self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
#         if self.batchnorm:
#             self.bn_main = nn.BatchNorm2d(out_channels)
#         if self.act:
#             self.relu = nn.ReLU()
#
#     def __repr__(self):
#         if hasattr(self, 'in_channels_remain') and hasattr(self, 'out_channels_remain'):
#             s = '\n(1): Conv_dhp({}, {}, kernel_size=({}, {}), stride={})'.format(
#                 self.in_channels_remain, self.out_channels_remain, self.kernel_size, self.kernel_size, self.stride)
#         else:
#             s = '\n(1): Conv_dhp({}, {}, kernel_size=({}, {}), stride={})'.format(
#                 self.in_channels, self.out_channels, self.kernel_size, self.kernel_size, self.stride)
#         if self.batchnorm:
#             s = s + '\n(2): ' + repr(self.bn_main)
#         if self.act:
#             s = s + '\n(3): ' + repr(self.relu)
#         s = addindent(s, 2)
#         return s
#
#     def set_parameters(self, vector_mask, calc_weight=False):
#         latent_vector_input, mask_input, mask_output = vector_mask
#         mask_input = mask_input.to(torch.float32).nonzero().squeeze(1)
#         mask_output = mask_output.to(torch.float32).nonzero().squeeze(1)
#         if calc_weight:
#             latent_vector_input = torch.index_select(latent_vector_input, dim=0, index=mask_input)
#             latent_vector_output = torch.index_select(self.latent_vector, dim=0, index=mask_output)
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
#         self.in_channels_remain = mask_input.shape[0]
#         self.out_channels_remain = mask_output.shape[0]
#
#         if self.batchnorm:
#             self.bn_main.num_features = mask_output.shape[0]
#             self.bn_main.num_features_remain = mask_output.shape[0]
#         if self.act:
#             self.relu.channels = mask_output.shape[0]
#
#     def calc_weight(self, latent_vector_input, latent_vector_output=None, mask_input=None, mask_output=None):
#         if latent_vector_output is None:
#             latent_vector_output = self.latent_vector
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
#         # embed()
#         weight = torch.matmul(latent_vector_input.unsqueeze(-1), latent_vector_output.unsqueeze(0)) + bias0
#         weight = weight.unsqueeze(-1) * weight1 + bias1
#         weight = torch.matmul(weight2, weight.unsqueeze(-1)).squeeze(-1) + bias2
#         # if weight.nelement() != self.in_channels * self.out_channels * self.kernel_size ** 2:
#         #     embed()
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
#         x, latent_vector_input = x
#
#         # execute the hypernetworks to get the weights of the backbone network
#         weight = self.calc_weight(latent_vector_input)
#
#         out = F.conv2d(x, weight, bias=self.bias, stride=self.stride, padding=self.kernel_size // 2)
#         return out
#
#     def forward_finetuning(self, input):
#         out = F.conv2d(input, self.weight, bias=self.bias, stride=self.stride, padding=self.kernel_size // 2)
#         return out