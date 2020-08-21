import torch.nn as nn
import torch
from model import common
from IPython import embed

def make_model(args, parent=False):
    return SRRESNET_CLUSTER(args)


class SRRESNET_CLUSTER(nn.Module):
    def __init__(self, args, conv3x3=common.default_conv,  conv1x1=common.default_conv):
        super(SRRESNET_CLUSTER, self).__init__()

        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        scale = args.scale[0]
        act_res_flag = args.act_res

        kernel_size = 3
        if act_res_flag == 'No':
            act_res = None
        else:
            act_res = 'prelu'

        head = [common.default_conv(args.n_colors, n_feats, kernel_size=9), nn.PReLU()]
        body = [common.ResBlock(conv3x3, n_feats, kernel_size, bn=True, act=act_res) for _ in range(n_resblock)]
        body.extend([conv3x3(n_feats, n_feats, kernel_size), nn.BatchNorm2d(n_feats)])

        tail = [
            common.Upsampler(conv3x3, scale, n_feats, act=act_res),
            conv3x3(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

        if conv3x3 == common.default_conv:
            # print('Loading from checkpoint {}'.format(args.pretrain_cluster))
            # for (k1, v1), (k2, v2) in zip(self.state_dict().items(), torch.load(args.pretrain_cluster).items()):
            #     print('{:<50}\t{:<50}\t{} {}'.format(k1, k2, list(v1.shape), list(v2.shape)))
            self.load_state_dict(torch.load(args.pretrain_cluster))
        # embed()
    def forward(self, x):
        x = self.head(x)
        f = self.body(x)
        x = self.tail(x + f)
        return x
