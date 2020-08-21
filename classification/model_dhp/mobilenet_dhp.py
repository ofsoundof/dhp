'''
This module applies diffrentiable pruning via hypernetworks to MobileNetV1.

MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_dhp.dhp_base import DHP_Base, conv_dhp


def make_model(args, parent=False):
    return MobileNet_DHP(args)


class DepthwiseSeparableConv_dhp(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1, embedding_dim=8):
        super(DepthwiseSeparableConv_dhp, self).__init__()
        self.finetuning = False
        self.layer1 = conv_dhp(in_planes, in_planes, kernel_size=3, stride=stride, groups=in_planes, embedding_dim=embedding_dim)
        self.layer2 = conv_dhp(in_planes, out_planes, kernel_size=1, stride=1, embedding_dim=embedding_dim)

    def set_parameters(self, vector_mask, calc_weight=False):
        self.layer1.set_parameters(vector_mask[:3], calc_weight)
        self.layer2.set_parameters(vector_mask[:2] + [vector_mask[3]], calc_weight)

    def forward(self, x):
        if not self.finetuning:
            x, latent_vector_input = x
            # embed()
            out = self.layer1([x, latent_vector_input])
            out = self.layer2([out, latent_vector_input])
        else:
            out = self.layer1(x)
            out = self.layer2(out)
        return out


class MobileNet_DHP(DHP_Base):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, args):
        super(MobileNet_DHP, self).__init__(args=args)
        self.width_mult = args.width_mult
        self.prune_classifier = args.prune_classifier
        self.linear_percentage = args.linear_percentage

        self.latent_vector_stage0 = nn.Parameter(torch.randn((3)))

        stride = 1 if args.data_train.find('CIFAR') >= 0 else 2
        self.features = nn.ModuleList([conv_dhp(3, int(32 * self.width_mult), kernel_size=3, stride=stride, embedding_dim=self.embedding_dim)])
        self.features.extend(self._make_layers(in_planes=int(32 * self.width_mult)))
        self.linear = nn.Sequential(nn.Dropout(0.2), nn.Linear(int(1024 * self.width_mult), self.n_classes))
        self.show_latent_vector()

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            out_planes = int(out_planes * self.width_mult)
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(DepthwiseSeparableConv_dhp(in_planes, out_planes, stride, embedding_dim=self.embedding_dim))
            in_planes = out_planes
        return layers

    def mask(self):
        masks = []
        for i, v in enumerate(self.gather_latent_vector(grad_prune=self.grad_prune, grad_normalize=self.grad_normalize)):
            if not self.prune_classifier:
                flag = i != 27 and not (i % 2 == 0)
            else:
                flag = not (i % 2 == 0)

            if flag:
                if i == 27:
                    channels = self.remain_channels(v, percentage=self.linear_percentage)
                else:
                    channels = self.remain_channels(v)

                # embed()
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
                flag = i != 27 and not (i % 2 == 0)
            else:
                flag = not (i % 2 == 0)
            if flag:
                if i == 27:
                    channels = self.remain_channels(v, percentage=self.linear_percentage)
                else:
                    channels = self.remain_channels(v)
                if torch.sum(v.abs() >= self.pt) > channels:
                    self.soft_thresholding(v, regularization)

    def set_parameters(self, calc_weight=False):
        latent_vectors, masks = self.gather_latent_vector(), self.mask()

        for i, layer in enumerate(self.features):
            if i == 0:
                vm = [latent_vectors[0], masks[0], masks[1]]
            else:
                vm = [latent_vectors[i * 2 - 1]] + masks[i * 2 - 1: i * 2 + 2]
            layer.set_parameters(vm, calc_weight)
        if self.prune_classifier:
            mask = self.mask()[-1]
            mask_input = mask.to(torch.float32).nonzero().squeeze(1)
            self.linear[1].in_features_remain = mask_input.shape[0]
            if calc_weight:
                self.linear[1].in_features = mask_input.shape[0]
                weight = self.linear[1].weight.data
                weight = torch.index_select(weight, dim=1, index=mask_input)
                self.linear[1].weight = nn.Parameter(weight)

    def forward(self, x):
        if not self.finetuning:
            latent_vectors = self.gather_latent_vector()
            for i, layer in enumerate(self.features):
                if i == 0:
                    x = layer([x, latent_vectors[0]])
                else:
                    # embed()
                    x = layer([x, latent_vectors[i * 2 - 1]])
        else:
            for i, layer in enumerate(self.features):
                x = layer(x)
        out = x.mean([2, 3])
        # out = F.avg_pool2d(x, 2)
        # out = out.view(out.size(0), -1)
        out = self.linear(out)
        # embed()
        return out


#
# def test():
#     net = MobileNet_DHP()
#     x = torch.randn(1,3,32,32)
#     y = net(x)
#     print(y.size())

# for i, (k, v) in enumerate(zip(kk, latent_vectors)):
#     print(i, list(v.shape), k)
