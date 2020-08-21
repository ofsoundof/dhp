import os
import math
from decimal import Decimal

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

class PWD(nn.Module):
    '''
        Calculating pair-wise distances between two tensors.
    '''
    def __init__(self):
        super(PWD, self).__init__()

    def forward(self, p, pn, c, c_sq):
        p_rep = pn.repeat(1, len(c))     # (n / b) x k
        c_rep = c_sq.repeat(len(p), 1)     # (n / b) x k
        pc = torch.mm(p, c.t()).mul_(2)    # (n / b) x k
        # from IPython import embed; embed()
        dist = p_rep + c_rep - pc
        min_dists, min_labels = dist.min(1)

        return min_dists, min_labels

class Clustering():
    def __init__(
        self, k,
        n_init=1, max_iter=4500, symmetry='i', cpu=False, n_GPUs=1):
        '''
            input arguments
                n_init:
                    number of different trials for k-means clustering

                max_iter:
                    maximum iteration for k-means clustering

                symmetry:
                    transform invariant clustering parameter

                    i           default k-means clustering
                    ih          horizontal flip invariant clustering
                    ihvo        flip invariant clustering
                    ihvoIHVO    flip and rotation invariant clustering

                cpu and n_GPUs:
                    you can use multiple GPUs for k-means clustering
        '''

        self.k = k
        self.n_init = n_init
        self.max_iter = max_iter
        self.symmetry = symmetry
        self.device = torch.device('cpu' if cpu else 'cuda')
        if not cpu:
            self.n_GPUs = n_GPUs
            self.pairwise_dist = nn.DataParallel(PWD(), range(n_GPUs))

        self.tf_dict = {k: v for v, k in enumerate('ihvoIHVO')}

    def fit(self, points):
        points = points.to(self.device)
        n, n_feats = points.size()
        self.make_sampler(n_feats, self.k)

        print('# points: {} / # parameters: {} / # clusters: {}'.format(
            n, points.nelement(), self.k)
        )

        print('Using {} initial seeds'.format(self.n_init))
        tqdm.monitor_interval = 0
        tqdm_init = tqdm(range(self.n_init), ncols=80)
        best = 1e8
        for _ in tqdm_init:
            tqdm_init.set_description('Best cost: {:.4f}'.format(best))
            with torch.no_grad():
                centroids, labels, cost = self.cluster(points)
            if cost < best:
                self.cluster_centers_ = centroids.clone()
                self.labels_ = labels.clone()
                best = cost

        print('')

        return self.cluster_centers_.cpu(), self.labels_.cpu()

    def cluster(self, points, log=False):
        n, n_feats = points.size()
        s = len(self.symmetry)

        labels = torch.LongTensor(n).to(self.device)
        ones = points.new_ones(n)
        init_seeds = ones.multinomial(self.k, replacement=False)
        # from IPython import embed; embed()
        pn = points.pow(2).sum(1, keepdim=True)
        centroids = points.index_select(0, init_seeds)
        tqdm_cl = tqdm(range(self.max_iter), ncols=80)

        # to prevent out of memory...
        if self.n_GPUs == 1:
            mem_check = self.k * n * n_feats
            mem_bound = 3 * 10**8
            if mem_check > mem_bound:
                split = round(mem_check / mem_bound)
            else:
                split = 1
        else:
            split = self.n_GPUs

        for _ in tqdm_cl:
            centroids_full = self.transform(centroids.repeat(s, 1))
            centroids_full = centroids_full.repeat(split, 1)
            cn = centroids_full.pow(2).sum(1)
            if self.n_GPUs == 1:
                min_dists = []
                min_labels = []
                for _p, _pn, _c, _cn in zip(
                    points.chunk(split),
                    pn.chunk(split),
                    centroids_full.chunk(split),
                    cn.chunk(split)):

                    md, ml = self.pairwise_dist(_p, _pn, _c, _cn)
                    min_dists.append(md)
                    min_labels.append(ml)

                min_dists = torch.cat(min_dists)
                min_labels = torch.cat(min_labels)
            else:
                min_dists, min_labels = self.pairwise_dist(
                    points, pn, centroids_full, cn
                )

            cost = min_dists.mean().item()
            change = (min_labels != labels).sum().item()
            if change == 0: break

            tqdm_cl.set_description(
                'C: {:.3e} / Replace {}'.format(cost, change)
            )

            centroids_new = points.new_zeros(s * self.k, n_feats)
            centroids_new.index_add_(0, min_labels, points)
            centroids_new = self.transform(centroids_new, inverse=True)
            centroids = sum(centroids_new.chunk(s))

            counts_new = points.new_zeros(s * self.k)
            counts_new.index_add_(0, min_labels, ones)
            counts = sum(counts_new.chunk(s))

            centroids.div_(counts.unsqueeze(-1))
            labels.copy_(min_labels)

        return centroids, labels, cost

    def make_sampler(self, n_feats, k):
        '''
            pre-define sampling grids to reduce overhead
            see get_grid() for details
        '''
        grid = get_grid(n_feats)
        grid_inverse = grid[[0, 1, 2, 3, 7, 5, 6, 4]]

        n_tfs = len(self.symmetry)
        list_tf = [[self.tf_dict[tf]] for tf in self.symmetry]
        idx = torch.LongTensor(list_tf).repeat(1, k).view(-1)
        offset = n_feats * torch.arange(n_tfs * k).long().view(-1, 1)
        # from IPython import embed; embed()
        self.sampler = (grid[idx] + offset).to(self.device)
        self.isampler = (grid_inverse[idx] + offset).to(self.device)

    def transform(self, kernels, inverse=False):
        '''
            forward/backward transform using the sampling grids
        '''
        flatten = kernels.view(-1)
        if not inverse:
            sampler = self.sampler
        else:
            sampler = self.isampler

        kernels_tf = flatten[sampler.view(-1)].view(kernels.size())

        return kernels_tf

def to_image(t):
    pos = t.clamp(min=0)
    neg = -t.clamp(max=0)
    zero = t.mul(0)

    return torch.cat((pos, zero, neg), dim=1)

def save_kernels(t, filename, highlight=False):
    k_kernels, kh, kw = t.size()
    t = t.view(k_kernels, -1)
    t = t / t.abs().max(dim=1, keepdim=True)[0]
    t = to_image(t.view(-1, 1, kh, kw))
    if highlight:
        output_padding = 0
        p, s = 2, 10
        t = F.upsample(t, scale_factor=s)
        t = F.pad(t, [p, p, p, p], value=255)
        t[0, [0, 2], (p - 1), (p - 1):(kw * s) + p + 1] = 0
        t[0, [0, 2], (kh * s) + p, (p - 1):(kw * s) + p + 1] = 0
        t[0, [0, 2], (p - 1):(kh * s) + p + 1, (p - 1)] = 0
        t[0, [0, 2], (p - 1):(kh * s) + p + 1, (kw * s) + p] = 0
    else:
        output_padding = 1

    if k_kernels <= 256:
        output_row = 16
    elif k_kernels <= 1024:
        output_row = 32
    else:
        output_row = 64

    torchvision.utils.save_image(
        t.cpu(),
        filename,
        nrow=output_row,
        padding=output_padding,
        pad_value=255
    )

def save_distribution(labels, filename):
    max_label = labels.max().item() + 1
    distribution = np.zeros(max_label)
    axis = np.linspace(1, max_label, max_label)
    for i in range(max_label):
        distribution[i] = labels.eq(i).sum()

    prob = distribution / distribution.sum()
    entropy = np.sum(-prob * np.log2(prob))
    entropy_ref = np.log2(max_label)

    plt.title('Entropy: {:.3f} / {:.3f}'.format(entropy, entropy_ref))
    plt.xlabel('Centroid index')
    plt.grid(True)
    plt.plot(axis, distribution)
    plt.savefig(filename)
    plt.close()

def get_grid(n_feats):
    side = int(math.sqrt(n_feats))
    '''
        make sampling grids for various transformations
    '''
    def _hflip(g):
        chunks = g.chunk(side, dim=1)
        return torch.cat([c for c in chunks[::-1]], dim=1)

    def _vflip(g):
        chunks = g.chunk(side, dim=0)
        return torch.cat([c for c in chunks[::-1]], dim=0)

    def _rot90(g):
        chunks = g.chunk(side, dim=0)
        return torch.cat([c.t() for c in chunks[::-1]], dim=1)

    base = torch.arange(n_feats).long().view(side, side)
    ls = [base]
    for f in _hflip, _vflip, _rot90:
        ls.extend([f(g) for g in ls])

    grid = torch.stack([g.view(-1) for g in ls], dim=0)

    return grid

def sort_clustering(kernels, indices, k_kernels):
    h = indices.float().histc(bins=k_kernels, min=0, max=k_kernels-1)
    _, perm = h.sort(descending=True)
    indices_perm = torch.zeros_like(indices)

    for i in range(k_kernels):
        indices_perm += i * (indices == perm[i]).long()

    indices = indices_perm
    kernels = kernels[perm]

    return kernels, indices

if __name__ == '__main__':
    s = torch.load('../../models/resnet18-c64.pt')
    kernels = s['centroids_1']
    save_kernels(kernels, 'resnet18-c64.png')
    with open('idx.txt', 'w') as f:
        idx = s['features.2.1.body.3.idx'].view(64, 64)
        for i in idx:
            for j in i:
                f.write('{} '.format(j.item()))

            f.write('\n')

