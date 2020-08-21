# This module is inherited from Shuhang. Used for Denoising.
import os
from data import common
import numpy as np
import imageio
import torch.utils.data as data
from IPython import embed
class ImageData(data.Dataset):
    def __init__(self, args, train=True, benchmark=False):
        self.args = args
        self.train = train
        self.split = 'train' if train else 'test'
        self.benchmark = benchmark
        self.scale = args.scale[0]
        self.sigma = args.noise_sigma
        self.quality = args.quality
        # from IPython import embed; embed();
        self._set_filesystem(args.dir_data)

        def _load_benchmark_bin():
            #self.images_tar = np.load(self._name_tarbin())*self.args.rgb_range
            #self.images_input = np.load(self._name_inputbin())*self.args.rgb_range
            self.images_tar = np.load(self._name_tarbin(), allow_pickle=True) * 255
            self.images_input = np.load(self._name_inputbin(), allow_pickle=True) * 255

        def _load_bin():
            self.images_tar = np.load(self._name_tarbin(), allow_pickle=True)
            self.images_input = np.load(self._name_inputbin(), allow_pickle=True)
        #from IPython import embed; embed();
        if self.benchmark:
            _load_benchmark_bin()
        elif args.ext == 'img':
            self.images_tar, self.images_input = self._scan()
        elif args.ext.find('sep') >= 0:
            print('Scaning')
            self.images_tar, self.images_input = self._scan()
            #print(self.images_input)
            print('Scan finished!')

            if args.ext.find('reset') >= 0:
                print('Preparing seperated binary files')
                for v in self.images_tar:
                    img_tar = imageio.imread(v)
                    name_sep = v.replace(self.ext, '.npy')
                    np.save(name_sep, img_tar)
                    print(name_sep)
                for v in self.images_input:
                    img_input = imageio.imread(v)
                    name_sep = v.replace(self.ext, '.npy')
                    np.save(name_sep, img_input)
                    print(name_sep)
            if self.ext == '.png':
                # embed()
                self.images_tar = [v.replace(self.ext, '.npy') for v in self.images_tar]
                self.images_input = [v.replace(self.ext, '.npy') for v in self.images_input]

        elif args.ext.find('bin') >= 0:
            try:
                if args.ext.find('reset') >= 0:
                    raise IOError
                print('Loading a binary file')
                _load_bin()
            except:
                print('Preparing a binary file')
                bin_path = os.path.join(self.apath, 'bin')
                if not os.path.isdir(bin_path):
                    os.mkdir(bin_path)

                list_tar, list_input = self._scan()
                img_tar = [imageio.imread(f) for f in list_tar]
                np.save(self._name_tarbin(), img_tar)
                del img_tar

                img_input = [imageio.imread(f) for f in list_input]
                np.save(self._name_inputbin(), img_input)
                del img_input

                _load_bin()
        else:
            print('Please define data type')

    def _scan(self):
        raise NotImplementedError

    def _set_filesystem(self, dir_data):
        raise NotImplementedError

    def _name_tarbin(self):
        raise NotImplementedError

    def _name_inputbin(self, scale):
        raise NotImplementedError

    def __getitem__(self, idx):
        #print(idx)
        img_input, img_tar, filename = self._load_file(idx)
        #print('aaaaaaaaaaaaaaaaaaaaaa:', img_input.shape, img_tar.shape)
        img_input, img_tar = common.set_channel([img_input, img_tar], self.args.n_colors)
        #print('bbbbbbbbbbbbbbbbbbbbb:', img_input.shape, img_tar.shape)
        img_input, img_tar = self._get_patch(img_input, img_tar)
        #print('ccccccccccccccccccc:', img_input.shape, img_tar.shape)
        input_tensor, tar_tensor = common.np2Tensor([img_input, img_tar], self.args.rgb_range)
        #print('dddddddddddddddddddddd:', input_tensor[0,0,0], tar_tensor[0,0,0])
        return input_tensor, tar_tensor, filename

    def __len__(self):
        return len(self.images_tar)

    def _get_index(self, idx):
        return idx

    def _load_file(self, idx):

        idx = self._get_index(idx)
        #print(idx)
        #print('total length {}, {}'.format(len(self.images_input), len(self.images_tar)))
        if self.benchmark:
            img_input = self.images_input[idx]
            img_tar = self.images_tar[idx]
            filename = str(idx + 1)
        elif self.args.ext == 'img':

            img_input = imageio.imread(self.images_input[idx])
            img_tar = imageio.imread(self.images_tar[idx])
            filename = self.images_tar[idx]
        elif self.args.ext.find('sep') >= 0:
            #print(self.images_input[idx])
            #print(idx)
            #print(self.images_tar[idx])
            img_input = np.load(self.images_input[idx])
            img_tar = np.load(self.images_tar[idx])
            filename = self.images_tar[idx]
            #print('aaaaaaaaaaaaaaaaaaaaaa:', img_input.shape, img_tar.shape)
        else:
            img_input = self.images_input[idx]
            img_tar = self.images_tar[idx]
            filename = str(idx + 1)
        return img_input, img_tar, filename

    def _get_patch(self, img_input, img_tar):
        patch_size = self.args.patch_size
        scale = self.scale
        #print('bbbbbbbbbbbbbbbbbbbbbbbb:', img_input.shape, img_tar.shape)
        if self.train:
            img_input, img_tar = common.get_patch(img_input, img_tar, patch_size, scale)
            #print('bbbbbbbbbbbbbbbbbbbbbbbb:', img_input.shape, img_tar.shape)
            img_input, img_tar = common.augment([img_input, img_tar])
            img_input = common.add_noise_shuhang(img_input, self.sigma)
            #print('ccccccccccccccccccccccc:', img_input.shape, img_tar.shape)
        else:
            ih, iw = img_input.shape[0:2]
            img_tar = img_tar[0:ih * scale, 0:iw * scale]

        return img_input, img_tar

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

