import math
from importlib import import_module
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np
import sklearn.cluster as skc

def gen_prune(target):
    return (
        m for m in target.modules() \
        if hasattr(m, 'weight') \
        and type(m).__name__.find('BatchNorm') == -1)

def register_cut(target, prune_type='kernel'):
    def _prune(layer, input):
        layer.weight.data *= layer.cut

    gen = gen_prune(target)
    for m in gen:
        w = m.weight.data
        if isinstance(m, nn.modules.conv._ConvNd):
            c_out, c_in, _, _ = w.size()
            if prune_type.find('channel') >= 0:
                cut = torch.ones(c_out, 1, 1, 1)
            elif prune_type.find('kernel') >= 0:
                cut = torch.ones(c_out, c_in, 1, 1)
            elif prune_type.find('weight') >= 0:
                cut = torch.ones(w.size())
        else:
            cut = torch.ones(w.size())
        # from IPython import embed; embed(); exit()
        m.register_buffer('cut', cut)
        m.register_forward_pre_hook(_prune)
    
def make_model(args, parent=False):
    module = import_module('model.' + args.base)

    class PruneNet(getattr(module, args.base)):
        def __init__(self, args, parent):
            super(PruneNet, self).__init__(args)
            self.prune_type = args.prune_type
            self.prune_1x1 = args.prune_type.find('1x1') >= 0
            self.prune_fc = args.prune_type.find('fc') >= 0

            if args.extend != '.' and not parent:
                self.load_state_dict(torch.load(args.extend))

            distribution = torch.cat([
                m.weight.data.abs().view(-1) \
                for m in gen_prune(self)]).sort()[0]
            n_weight = distribution.nelement()

            print('Minimum weight: {:.4f}'.format(distribution[0]))
            for i in range(1, 10):
                print('{}%: {:.4f}'.format(
                    i * 10, distribution[i * (n_weight // 10)]))
            print('Maximum weight: {:.4f}'.format(distribution[-1]))

            register_cut(self, args.prune_type)

        def prune(self, threshold):
            total = 0
            left = 0
            total_1x1 = 0
            left_1x1 = 0
            total_3x3 = 0
            left_3x3 = 0
            total_fc = 0
            left_fc = 0

            for m in gen_prune(self):
                is_conv = isinstance(m, nn.modules.conv._ConvNd)
                w = m.weight.data
                do_prune = True
                if is_conv:
                    c_out, c_in, kh, kw = w.size()
                    if self.prune_type.find('channel') >= 0:
                        energy = w.view(c_out, -1).pow(2).sum(1).sqrt()
                        n_cuts = c_out
                    elif self.prune_type.find('kernel') >= 0:
                        w = w.view(c_out * c_in, -1)
                        if self.prune_type.find('L1') >= 0:
                            energy = w.abs().sum(1)
                        else:
                            energy = w.pow(2).sum(1).sqrt()
                        #energy.mul_(c_in)
                        n_cuts = c_out * c_in
                    elif self.prune_type.find('weight') >= 0:
                        energy = w.view(-1).abs()
                        n_cuts = c_out * c_in * kh * kw

                    if kh * kw == 1:
                        if self.prune_1x1:
                            threshold = 0.1
                        else:
                            do_prune = False
                else:
                    c_out, c_in = w.size()
                    energy = w.view(-1).abs()
                    n_cuts = c_out * c_in

                    if not self.prune_fc:
                        do_prune = False
                
                if do_prune:
                    if self.prune_type.find('percent') >= 0:
                        # We do not prune the inpuy layer
                        if w.size(1) == 3:
                            m.cut.fill_(1) 
                        else:
                            keep = int(n_cuts * (100 - threshold) // 100) + 1
                            t = energy.topk(keep)[0][-1]
                            m.cut.copy_(
                                energy.gt(t).view(m.cut.size()).float()
                            )
                    elif self.prune_type.find('threshold') >= 0:
                        m.cut.copy_(
                            energy.gt(threshold).view(m.cut.size()).float()
                        )
                
                ct = w.nelement()
                cl = w.nelement() * m.cut.sum() // n_cuts

                if is_conv:
                    if kh * kw > 1:
                        total_3x3 += ct
                        left_3x3 += cl
                    elif kh * kw == 1:
                        total_1x1 += ct
                        left_1x1 += cl
                else:
                    total_fc += ct
                    left_fc += cl
                    
                total += ct
                left += cl

            def _percent(l, t):
                if t != 0:
                    return [int(l), t, 100 * l / t]
                else:
                    return [0, 0, 0]

            res = '\n'.join([
            'Pruning results',
            '{}/{} connections left. ({:.2f}%)'.format(
                *_percent(left, total)),
            '{}/{} connections left (3x3 conv) ({:.2f}%)'.format(
                *_percent(left_3x3, total_3x3)),
            '{}/{} connections left (1x1 conv) ({:.2f}%)'.format(
                *_percent(left_1x1, total_1x1)),
            '{}/{} connections left (fully connected) ({:.2f}%)'.format(
                *_percent(left_fc, total_fc)),
            ''])

            return res
        
    return PruneNet(args, parent)

