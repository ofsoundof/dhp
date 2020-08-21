import os
import math
from importlib import import_module
from itertools import chain
from model import common
from model import cconv
from model import qnn
from model import clustering
from model import prunenet
from IPython import embed

import torch
import torch.nn as nn

def make_model(args, parent=False):
    module = import_module('model.' + args.base.lower())
    module_p = import_module('model.' + args.base_p.lower())
    print(args.base)
    print(args.base_p)
    conv1x1 = common.default_conv
    if args.symmetry == 'i':
        conv3x3 = cconv.cluster_conv
    else:
        conv3x3 = cconv.gscluster_conv

    if args.n_bits < 16:
        cconv.CConv.n_bits = args.n_bits
        qnn.QuantizeParams.bits_w = args.n_bits

    class ClusterNet(getattr(module, args.base)):
        def __init__(self, args):
            super(ClusterNet, self).__init__(
                args, conv3x3=conv3x3, conv1x1=conv1x1
            )

            # prepare the reference network
            parent = module_p.make_model(args, parent=True)
            prunenet.register_cut(parent)
            if args.extend:
                self.load_state_dict(
                    parent.state_dict(),
                    strict=False,
                    init=True
                )

            # count convs and params
            n_convs = sum(1 for m in self.gen(parent))

            n_params = sum(m.weight.nelement() for m in self.gen(parent))
            print('{} 3x3 convs / {} params.'.format(n_convs, n_params))

            # split set for multi clustering
            self.split_set(self.gen(parent), n_convs, args.multi)

            # extract all kernels from the parent network
            k_list, s_dict = self.extract(parent, args.scale_type)

            # clustering
            idx_list = self.get_centroids(
                k_list,
                args.n_init,
                args.max_iter,
                symmetry=args.symmetry,
                n_GPUs=args.n_GPUs
            )

            # replace kernels with centroids
            self.quantize_kernels(parent, idx_list, s_dict, debug=args.debug)

        def split_set(self, gen, n_convs, multi):
            '''
                multi clustering examples

                --multi full:
                    use a set of centroids for all kernels in the network

                --multi split:
                    use a set of centroids for each layer in the network

                --multi 0-0=64,1-16=256:
                    use 64 centroids for conv0 ~ conv0
                    use 256 centroids for conv1 ~ conv16

                    you can also try different number of centroids, different splits
            '''
            if multi.find('full') >= 0:
                self.split = {
                    'k_kernels': [int(multi.split('-')[-1])],
                    'map': {i: 0 for i in range(n_convs)}
                }
                # embed()
            elif multi == 'split':
                kernels = [m.in_channels * m.out_channels for m in gen]
                self.split = {
                    'k_kernels': [min(k, k_kernels) for k in kernels],
                    'map': {i: i for i in range(n_convs)}
                }
            else:
                self.split = {'k_kernels': [], 'map': {}}
                for mi, m in enumerate(multi.split(',')):
                    entry, k_kernels_split = m.split('=')
                    self.split['k_kernels'].append(int(k_kernels_split))
                    if entry.find('+') >= 0:
                        entry = [int(e) for e in entry.split('+')]
                    elif entry.find('-') >= 0:
                        lb, ub = [int(e) for e in entry.split('-')]
                        entry = [i for i in range(lb, ub + 1)]

                    for e in entry:
                        self.split['map'][e] = mi

        # def gen(self, target, conv1x1=False):
        #     '''
        #          a generator for iterating default Conv2d
        #     '''
        #     def _criterion(m):
        #         if isinstance(m, nn.Conv2d):
        #             if conv1x1:
        #                 return m.kernel_size[0] * m.kernel_size[1] == 1
        #             else:
        #                 return m.kernel_size[0] * m.kernel_size[1] > 1
        #         elif isinstance(m, nn.ConvTranspose2d):
        #             return True
        #
        #         return False
        #     if target.__class__.__name__ == 'EDSR_CLUSTER':
        #         target = target.body
        #         gen = (m for m in target.modules() if _criterion(m))
        #     elif target.__class__.__name__ == 'SRRESNET1_CLUSTER':
        #         # gen = (m for m in chain(target.body_r.modules(), target.body_conv.modules()) if _criterion(m))
        #         gen = (m for m in target.body_r.modules() if _criterion(m))
        #     else:
        #         gen = (m for m in target.modules() if _criterion(m))
        #     return gen
        #
        def gen(self, target, conv1x1=False):
            '''
                 a generator for iterating default Conv2d
            '''
            def _criterion(m):
                if isinstance(m, nn.Conv2d):
                    if conv1x1:
                        return m.kernel_size[0] * m.kernel_size[1] == 1
                    else:
                        return m.kernel_size[0] * m.kernel_size[1] > 1
                # elif isinstance(m, nn.ConvTranspose2d):
                #     return True

                return False

            if target.__class__.__name__ == 'SRRESNET_CLUSTER':
                # target = target.head
                gen = [m for m in target.body.modules() if _criterion(m)] + [m for m in target.tail.modules() if _criterion(m)]
            #     elif target.__class__.__name__ == 'SRRESNET1_CLUSTER':
            #         # gen = (m for m in chain(target.body_r.modules(), target.body_conv.modules()) if _criterion(m))
            else:
                gen = (m for m in target.modules() if _criterion(m))
            return gen

        def extract(self, parent, scale_type):
            k_list = [[] for _ in self.split['k_kernels']]
            s_dict = {}
            modules = [m for m in self.gen(parent)]

            for k, v in self.split['map'].items():
                kernels, scales = self.preprocess_kernels(
                    modules[k],
                    scale_type
                )
                k_list[v].extend(list(kernels))
                s_dict[k] = nn.Parameter(
                    scales, requires_grad=(scale_type.find('train') >= 0)
                )

            return k_list, s_dict

        def preprocess_kernels(self, m, scale_type):
            '''
                return
                    kernels:
                        normalized kernels

                    magnitudes:
                        scales
            '''
            c_out, c_in, kh, kw = m.weight.size()
            weights = m.weight.data.view(c_out, c_in, kh * kw)
            cuts = m.cut.view(c_out, c_in, 1).byte()

            if scale_type.find('kernel') >= 0:
                magnitudes = weights.norm(2, dim=2)             # c_out x c_in

                if scale_type.find('norm') >= 0:
                    magnitudes.mul_(weights[:, :, (kh * kw) // 2].sign())
                else:
                    magnitudes.fill_(1)

                magnitudes.unsqueeze_(-1)                       # c_out x c_in x 1
                kernels = weights.div(magnitudes)               # c_out x c_in x (kh * kw)
                magnitudes.mul_(cuts.float()).unsqueeze_(-1)    # c_out x c_in x 1 x 1
            # from IPython import embed; embed(); exit()
            kernels = kernels.masked_select(cuts.to(torch.bool)).view(-1, kh, kw)

            return kernels, magnitudes

        def get_centroids(
            self, k_list, n_init, max_iter, symmetry='i', n_GPUs=1):
            '''
                k-means clustering using custom module
                see clustering.py for details
            '''

            idx_list = []
            for i, kernels in enumerate(k_list):
                entry = [k for k, v in self.split['map'].items() if v == i]
                print('\nCluster {}'.format(i))
                print('Conv {}'.format(', '.join(str(e) for e in entry)))

                k_kernels = self.split['k_kernels'][i]
                # embed()
                kernel_stack = torch.stack(kernels)
                # from IPython import embed; embed();
                kh, kw = kernel_stack.size()[-2:]

                cluster = clustering.Clustering(
                    k_kernels,
                    n_init=n_init,
                    max_iter=max_iter,
                    symmetry=symmetry,
                    n_GPUs=n_GPUs
                )
                centroids, idx = cluster.fit(kernel_stack.view(-1, kh * kw))
                centroids = centroids.view(-1, kh, kw)
                idx_tic = idx.div(k_kernels)
                centroids, idx = clustering.sort_clustering(
                    centroids, idx % k_kernels, k_kernels
                )
                idx_list.append(idx + k_kernels * idx_tic)
                self.register_parameter(
                    'centroids_{}'.format(i), nn.Parameter(centroids)
                )
                self.save_clustering_results(i, kernel_stack, centroids, idx)

            return idx_list

        def save_clustering_results(self, i, kernel_stack, centroids, idx):
            _, kh, kw = kernel_stack.size()
            k_kernels = centroids.size(0)
            savedir = os.path.join(args.dir_save, args.save, 'clustering_results')
            # savedir = '/scratch_net/ofsoundof/yawli/srbasis/{}/clustering_results'.format(args.save)
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            clustering.save_kernels(
                centroids,
                '{}/iter000_c{:0>2}.png'.format(savedir, i)
            )
            clustering.save_distribution(
                idx,
                '{}/distribution_{:0>2}.pdf'.format(savedir, i)
            )
            torch.save(
                idx,
                '{}/labels_{:0>2}.pt'.format(savedir, i)
            )
            # visualize full kernels and their centroids
            if args.debug:
                for t in range(k_kernels):
                    mask = (idx == t)
                    mask_idx = torch.arange(len(kernel_stack))
                    # print('222222222222222222')
                    # embed()
                    mask_idx = mask_idx.masked_select(mask).long()

                    ref = centroids[t].view(1, kh, kw)
                    collect = torch.cat((ref, kernel_stack[mask_idx]), dim=0)
                    clustering.save_kernels(
                        collect,
                        '{}/vis_c{:0>2}_{:0>4}.png'.format(savedir, i, t),
                        highlight=True
                    )

        def quantize_kernels(
            self, parent, idx_list, s_dict, debug=False):

            modules_parent = [m for m in self.gen(parent)]
            modules_self = [m for m in cconv.gen_cconv(self)]
            # embed()
            idx_acc = [0] * len(self.split['k_kernels'])
            for k, v in self.split['map'].items():
                source = modules_parent[k]
                target = modules_self[k]
                c_out, c_in, _, _ = source.weight.size()
                idx = torch.LongTensor(c_out * c_in)
                for i, is_cut in enumerate(source.cut.view(-1)):
                    if is_cut != 0:
                        idx[i] = idx_list[v][idx_acc[v]]
                        idx_acc[v] += 1
                    # ignore pruned kernels
                    else:
                        idx[i] = 0

                target.set_params(
                    source,
                    centroids=getattr(self, 'centroids_{}'.format(v)),
                    idx=idx,
                    scales=s_dict[k],
                    debug=debug
                )
            # from IPython import embed; embed(); exit()

        def state_dict(self, destination=None, prefix='', keep_vars=False):
            state = super(ClusterNet, self).state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars)

            remove_key = []
            for k in state.keys():
                if k.find('centroids') >= 0 and k.find('_') == -1:
                    remove_key.append(k)

                if k.find('sampler') >= 0:
                    remove_key.append(k)

            for r in remove_key:
                state.pop(r)

            return state

        def load_state_dict(self, state_dict, strict=True, init=False):
            super(ClusterNet, self).load_state_dict(state_dict, strict=False)

            if not init:
                for i, m in enumerate(cconv.gen_cconv(self)):
                    m.kernels = getattr(
                        self,
                        'centroids_{}'.format(self.split['map'][i])
                    )

    return ClusterNet(args)

