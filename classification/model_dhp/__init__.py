import os
from importlib import import_module
import torch
import torch.nn as nn
# from IPython import embed

class Model(nn.Module):
    def __init__(self, args, checkpoint, converging=False):
        """
        :param args:
        :param checkpoint:
        :param converging: needed to decide whether to load the optimization or the finetune model
        """
        super(Model, self).__init__()
        print('Making model...')

        self.args = args
        self.ckp = checkpoint
        self.crop = args.crop
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.precision = args.precision
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models

        print('Import Module')
        module = import_module('model_dhp.' + args.model.lower())
        self.model = module.make_model(args)
        self.model = self.model.to(self.device)
        if args.precision == 'half':
            self.model = self.model.half()
        if not args.cpu:
            print('CUDA is ready!')
            torch.cuda.manual_seed(args.seed)
            if args.n_GPUs > 1:
                if not isinstance(self.model, nn.DataParallel):
                    self.model = nn.DataParallel(self.model, range(args.n_GPUs))

        # In Phase 2 (testing) or Phase 4 (loading for continuing)
        if args.test_only or (args.load and converging):
            self.get_model().reset_after_searching()

        self.load(pretrain=args.pretrain, load=args.load, resume=args.resume, cpu=args.cpu, converging=converging)

        if self.args.print_model and not args.test_only:
            print(self.get_model(), file=self.ckp.log_file)
            print(self.get_model())
        self.summarize(self.ckp)

    def forward(self, x):
        if self.crop > 1:
            b, n_crops, c, h, w = x.size()
            x = x.view(-1, c, h, w)
        x = self.model(x)

        if self.crop > 1: x = x.view(b, n_crops, -1).mean(1)

        return x

    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    def state_dict(self, **kwargs):
        return self.get_model().state_dict(**kwargs)

    def save(self, apath, epoch, converging=False, is_best=False):
        target = self.get_model().state_dict()

        conditions = (True, is_best, self.save_models)

        if converging:
            names = ('converging_latest', 'converging_best', 'converging_{}'.format(epoch))
        else:
            names = ('latest', 'best', '{}'.format(epoch))

        for c, n in zip(conditions, names):
            if c:
                torch.save(target, os.path.join(apath, 'model', 'model_{}.pt'.format(n)))

    def load(self, pretrain='', load='', resume=-1, cpu=False, converging=False):
        """
        Use pretrain and load to determine how to load the model.
        1. pretrain = '', load = ''. Training phase. Model loading is net needed during the training phase.
        2. pretrain = '**', load = ''. Testing phase.
        3. pretrain = '', load = '**'. Loading phase. Only need to provide the directory
        4. pretrain = '**', load = '**'. This is not valid.

        Phase 1: training, nothing is done here.
        Phase 2: testing, the pretrained pruned model is loaded.
        Phase 3: loading for searching, the saved unpruned model is loaded.
        Phase 4: loading for converging, the saved pruned model is loaded.
        """

        if not pretrain and not load:
            # Phase 1, training phase, do not load any model.
            f = None
            print('During training phase, no pretrained model is loaded.')
        elif pretrain and not load:
            # Phase 2, testing phase, loading the pruned model, strict = False.
            f = os.path.join(pretrain, 'model/model_latest.pt') if pretrain.find('.pt') < 0 else pretrain
            print('Load pre-trained model from {}'.format(f))
            strict = False
        elif not pretrain and load:
            # loading phase
            if not converging:
                # Phase 3, loading in the searching stage, loading the unpruned model, strict =True
                if resume == -1:
                    print('Load model after the last epoch')
                    resume = 'latest'
                else:
                    print('Load model after epoch {}'.format(resume))
                strict = True
            else:
                # Phase 4, loading in the converging stage, loading the pruned model, strict = False
                if resume == -1:
                    print('Load model after the last epoch in converging stage')
                    resume = 'converging_latest'
                else:
                    print('Load model after epoch {} in the converging stage'.format(resume))
                    resume = 'converging_{}'.format(resume)
                strict = False
            f = os.path.join(load, 'model', 'model_{}.pt'.format(resume))
        else:
            raise ValueError('args.pretrain and args.load should not be provided at the same time.')

        if f:
            kwargs = {}
            if cpu:
                kwargs = {'map_location': lambda storage, loc: storage}
            state = torch.load(f, **kwargs)
            print('First')
            self.get_model().load_state_dict(state, strict=strict)
            print('Second')

    def begin(self, epoch, ckp):
        self.train()
        m = self.get_model()
        if hasattr(m, 'begin'):
            m.begin(epoch, ckp)

    def log(self, ckp):
        m = self.get_model()
        if hasattr(m, 'log'): m.log(ckp)

    def summarize(self, ckp):
        ckp.write_log('# parameters: {:,}'.format(sum([p.nelement() for p in self.model.parameters()])))

        kernels_1x1 = 0
        kernels_3x3 = 0
        kernels_others = 0
        gen = (c for c in self.model.modules() if isinstance(c, nn.Conv2d))
        for m in gen:
            kh, kw = m.kernel_size
            n_kernels = m.in_channels * m.out_channels
            if kh == 1 and kw == 1:
                kernels_1x1 += n_kernels
            elif kh == 3 and kw == 3:
                kernels_3x3 += n_kernels
            else:
                kernels_others += n_kernels

        linear = sum([l.weight.nelement() for l in self.model.modules()  if isinstance(l, nn.Linear)])

        ckp.write_log('1x1: {:,}\n3x3: {:,}\nOthers: {:,}\nLinear:{:,}\n'.
                      format(kernels_1x1, kernels_3x3, kernels_others, linear), refresh=True)