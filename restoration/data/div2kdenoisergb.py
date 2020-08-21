import os
import glob
from data import common
from data import imagedata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class DIV2KDENOISERGB(imagedata.ImageData):
    def __init__(self, args, train=True):
        super(DIV2KDENOISERGB, self).__init__(args, train)

        # self.repeat = args.test_every // (args.n_train // args.batch_size)
        self.repeat = round(args.test_every / (args.n_train / args.batch_size))

    def _scan(self):
        list_tar = sorted(glob.glob(os.path.join(self.dir_tar, '*.png')))
        list_input = sorted(glob.glob(os.path.join(self.dir_input, '*.png')))

        # list_tar = []
        # list_input = []
        # if self.train:
        #     idx_begin = 0
        #     idx_end = self.args.n_train
        # else:
        #     idx_begin = self.args.n_train
        #     idx_end = self.args.offset_val + self.args.n_val
        #
        # for i in range(idx_begin + 1, idx_end + 1):
        #     filename = '{:0>4}'.format(i)
        #     list_tar.append(os.path.join(self.dir_tar, filename + self.ext))
        #     list_input.append(os.path.join(self.dir_input,filename + self.ext))

        return list_tar, list_input

    def _set_filesystem(self, dir_data):
        # from IPython import embed; embed(); exit()
        self.apath = os.path.dirname(os.path.dirname(dir_data)) + '/DIV2K/'
        self.dir_tar = os.path.join(self.apath, 'GT_sub')
        self.dir_input = os.path.join(self.apath, 'GT_sub')
        self.ext = '.png'

    def _name_tarbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.npy'.format(self.split)
        )

    def _name_inputbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.npy'.format(self.split)
        )

    def __len__(self):
        if self.train:
            return len(self.images_tar) * self.repeat
        else:
            return len(self.images_tar)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_tar)
        else:
            return idx


# class DIV2KDENOISERGB(ImageData.ImageData):
#     def __init__(self, args, train=True):
#         super(DIV2KDENOISERGB, self).__init__(args, train)
#         self.repeat = args.test_every // (args.n_train // args.batch_size)
#
#     def _scan(self):
#         list_tar = []
#         list_input = []
#         if self.train:
#             idx_begin = 0
#             idx_end = self.args.n_train
#         else:
#             idx_begin = self.args.n_train
#             idx_end = self.args.offset_val + self.args.n_val
#
#         for i in range(idx_begin + 1, idx_end + 1):
#             filename = '{:0>4}'.format(i)
#             list_tar.append(os.path.join(self.dir_tar, filename + self.ext))
#             list_input.append(os.path.join(self.dir_input,filename + self.ext))
#
#         return list_tar, list_input
#
#     def _set_filesystem(self, dir_data):
#         self.apath = dir_data + 'DIV2KRGB'
#         self.dir_tar = os.path.join(self.apath, 'Train_HR')
#         self.dir_input = os.path.join(self.apath, 'Train_HR')
#         self.ext = '.png'