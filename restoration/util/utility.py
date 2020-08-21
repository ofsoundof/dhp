import os
import math
import time
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import imageio
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        #self.img_sfx = args.model + '_' + args.submodel
        self.img_sfx = args.model


        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join(args.dir_save, args.save)
            if args.reset:
                os.system('rm -rf ' + self.dir)
        else:
            self.dir = os.path.join(args.dir_save, args.load)
            self.log = torch.load(self.dir + '/psnr_log.pt')
            print('Continue from epoch {}...'.format(len(self.log)))
            if not os.path.exists(self.dir):
                args.load = ''

        # if args.load == '.':
        #     if args.save == '.': args.save = now
        #     self.dir = os.path.join(args.dir_save, args.save)
        # else:
        #     self.dir = os.path.join(args.dir_save, args.save)
        #     if not os.path.exists(self.dir):
        #         args.load = '.'
        #     else:
        #         self.log = torch.load(self.dir + '/psnr_log.pt')
        #         print('Continue from epoch {}...'.format(len(self.log)))
        #
        # if args.reset:
        #     os.system('rm -rf ' + self.dir)
        #     args.load = '.'

        # def _make_dir(path):
        #     if not os.path.exists(path): os.makedirs(path)
        #
        # _make_dir(self.dir)
        # _make_dir(self.dir + '/model')
        # _make_dir(self.dir + '/results/' + self.args.data_test)

        os.makedirs(os.path.join(self.dir, 'model'), exist_ok=True)
        os.makedirs(os.path.join(self.dir, 'results', self.args.data_test), exist_ok=True)
        os.makedirs(os.path.join(self.dir, 'features'), exist_ok=True)
        os.makedirs(os.path.join(self.dir, 'per_layer_compression_ratio'), exist_ok=True)

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def save(self, trainer, epoch, converging=False, is_best=False):
        trainer.model.save(self.dir, epoch, converging=converging, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        torch.save(self.log, os.path.join(self.dir, 'psnr_log.pt'))

        if not converging:
            torch.save(trainer.optimizer.state_dict(), os.path.join(self.dir, 'optimizer.pt'))
            torch.save(epoch, os.path.join(self.dir, 'epochs.pt'))
        else:
            torch.save(trainer.optimizer.state_dict(), os.path.join(self.dir, 'optimizer_converging.pt'))
            torch.save(epoch, os.path.join(self.dir, 'epochs_converging.pt'))

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.array(range(len(self.log)))
        # axis = np.linspace(1, epoch, epoch)
        label = 'SR on {}'.format(self.args.data_test)
        fig = plt.figure()
        plt.title(label)
        for idx_scale, scale in enumerate(self.args.scale):
            plt.plot(
                axis,
                self.log[:, idx_scale].numpy(),
                label='Scale {}'.format(scale)
            )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(self.dir, self.args.data_test))
        plt.close(fig)

    def save_results(self, filename, save_list, scale):
        filename = '{}/results/{}/{}_x{}_'.format(self.dir, self.args.data_test, filename, scale)
        postfix = (self.img_sfx, 'LR', 'HR')
        for v, p in zip(save_list, postfix):
            normalized = v[0].data.mul(255 / self.args.rgb_range).clamp(-255, 255)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            imageio.imwrite('{}{}.png'.format(filename, p), ndarr)

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def calc_psnr(sr, hr, scale, rgb_range, div2k=False):
    diff = (sr - hr).data.div(rgb_range)
    if div2k:
        shave = scale + 6
    else:
        shave = scale
        if diff.size(1) > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    '''
    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    else:
        shave = scale + 6
    '''
    if shave == 0:
        valid = diff
    else:
        valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

def make_optimizer_dhp(args, my_model, ckp=None, lr=None, converging=False):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(args.optimizer))

    kwargs['lr'] = args.lr if lr is None else lr
    kwargs['weight_decay'] = args.weight_decay

    optimizer = optimizer_function(trainable, **kwargs)

    if args.load != '' and ckp is not None:
        if not converging:
            print('Loading optimizer in the searching stage from the checkpoint...')
            optimizer.load_state_dict(torch.load(os.path.join(ckp.dir, 'optimizer.pt')))
        else:
            print('Loading optimizer in the converging stage from the checkpoint...')
            if os.path.exists(os.path.join(ckp.dir, 'optimizer_converging.pt')):
                optimizer.load_state_dict(torch.load(os.path.join(ckp.dir, 'optimizer_converging.pt')))
    return optimizer


def make_scheduler_dhp(args, my_optimizer, decay, converging=False):

    if args.load != '':
        if not converging:
            last_epoch = torch.load(os.path.join(args.dir_save, args.save, 'epochs.pt'))
        else:
            if os.path.exists(os.path.join(args.dir_save, args.save, 'epochs_converging.pt')):
                last_epoch = torch.load(os.path.join(args.dir_save, args.save, 'epochs_converging.pt'))
            else:
                last_epoch = -1
    else:
        last_epoch = -1

    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=decay,
            gamma=args.gamma,
            last_epoch=last_epoch
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma,
            last_epoch=last_epoch
        )
    else:
        raise NotImplementedError('Scheduler type {} is not implemented'.format(args.decay_type))

    return scheduler


def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable, **kwargs)

def make_scheduler(args, my_optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )

    return scheduler