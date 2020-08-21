import os

from data import common
from data import imagedata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data
from IPython import embed
class BenchmarkDenoise(imagedata.ImageData):
    def __init__(self, args, train=True):
        super(BenchmarkDenoise, self).__init__(args, train, benchmark=True)

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'denoise/BenchmarkDenoise', self.args.data_test)
        # from IPython import  embed; embed(); exit()

    def _name_tarbin(self):
        # embed()
        if 'DIV2K100Color' in self.apath:
            return os.path.join(
                self.apath,
                'bin',
                'DIV2KColor_bin_tar.npy'
            )
        elif 'DIV2K100Gray' in self.apath:
            return os.path.join(
                self.apath,
                'bin',
                'DIV2KGray_bin_tar.npy'
            )
        else:
            return os.path.join(
                self.apath,
                'bin',
                '{}_bin_tar.npy'.format(self.split)
            )

    def _name_inputbin(self):
        if 'DIV2K100Color' in self.apath:
            return os.path.join(
                self.apath,
                'bin',
                'DIV2KColor_sigma{}_bin.npy'.format(self.sigma)
            )
        elif 'DIV2K100Gray' in self.apath:
            return os.path.join(
                self.apath,
                'bin',
                'DIV2KGray_sigma{}_bin.npy'.format(self.sigma)
            )
        else:
            return os.path.join(
                self.apath,
                'bin',
                '{}_bin_x{}_sigma{}.npy'.format(self.split, self.scale, self.sigma)
            )
