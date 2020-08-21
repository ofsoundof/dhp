import os
from importlib import import_module
import torch
import torch.nn as nn
# from IPython import embed


class Model(nn.Module):
    def __init__(self, args, ckp, converging=False):
        super(Model, self).__init__()
        print('Making model...')

        self.ckp = ckp
        self.args = args
        self.scale = args.scale
        self.idx_scale = 0
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models

        self.model_flag = args.model
        self.normal_lr = args.normal_lr == 'lr'
        self.input_res = args.input_res

        print('Import Module')
        module = import_module('model_dhp.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)
        if args.precision == 'half':
            self.model.half()
        if not args.cpu and args.n_GPUs > 1:
            print('CUDA is ready!')
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))

        # In Phase 2 (testing) or Phase 4 (loading for continuing)
        if args.test_only or (args.load and converging):
            self.get_model().reset_after_searching()

        self.load(pretrain=args.pretrain, load=args.load, resume=args.resume, cpu=args.cpu, converging=converging)

        if args.print_model and not args.test_only:
            print(self.get_model(), file=self.ckp.log_file)
            print(self.model)
        self.summarize(self.ckp)

    def forward(self, idx_scale, *x):
        self.idx_scale = idx_scale
        target = self.get_model()
        if hasattr(target, 'set_scale'):
            target.set_scale(idx_scale)
        # from IPython import embed; embed(); exit()

        if self.self_ensemble and not self.training:
            if self.chop:
                forward_function = self.forward_chop
            else:
                forward_function = self.model.forward

            return self.forward_x8(x, forward_function)
        elif self.chop and not self.training:
            return self.forward_chop(x)
        else:
            if self.model_flag.lower() == 'finetune':
                return self.model(x)
            else:
                # print('seventh')
                return self.model(x[0])

    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, apath, epoch, is_best=False, converging=False):
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
            self.get_model().load_state_dict(state, strict=strict)

    def forward_chop(self, x, shave=10, min_size=160000):
        scale = self.scale[self.idx_scale]
        n_GPUs = min(self.n_GPUs, 4)
        # from IPython import embed; embed();
        # if len(x[0].size()) == 3:
        #     c, h, w = x[0].size()
        # else:
        b, c, h, w = x[0].size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave

        def _patch_division(x, h_size, w_size, h, w):
            x_list = [
                x[:, :, 0:h_size, 0:w_size],
                x[:, :, 0:h_size, (w - w_size):w],
                x[:, :, (h - h_size):h, 0:w_size],
                x[:, :, (h - h_size):h, (w - w_size):w]]
            if len(x_list[0].size()) == 3:
                x_list = [torch.unsqueeze(x, 0) for x in x_list]
            # from IPython import embed; embed();
            return x_list


        if self.model_flag.lower() == 'finetune':
            lr_list = _patch_division(x[0], h_size, w_size, h, w)
            if self.normal_lr or self.input_res == 'hr':
                nl_list = _patch_division(x[1], h_size, w_size, h, w)
            else:
                nl_list = _patch_division(x[1], h_size * scale, w_size * scale, h * scale, w * scale)
        else:
            lr_list = _patch_division(x[0], h_size, w_size, h, w)
        # from IPython import embed; embed();
        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                if self.model_flag.lower() == 'finetune':
                    nl_batch = torch.cat(nl_list[i:(i + n_GPUs)], dim=0)
                    sr_batch = self.model((lr_batch, nl_batch))
                else:
                    sr_batch = self.model(lr_batch)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            if self.model_flag.lower() == 'finetune':
                sr_list = [
                    self.forward_chop(patch, shave=shave, min_size=min_size) \
                    for patch in zip(lr_list, nl_list)
                ]
            else:
                sr_list = [
                    self.forward_chop((patch,), shave=shave, min_size=min_size) \
                    for patch in lr_list
                ]
        if self.input_res == 'lr':
            h, w = scale * h, scale * w
            h_half, w_half = scale * h_half, scale * w_half
            h_size, w_size = scale * h_size, scale * w_size
            shave *= scale

        output = x[0].new(b, c, h, w)# if self.model_flag.lower() == 'finetune' else x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def forward_x8(self, x, forward_function):
        def _transform(v, op):
            if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [x]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])

        sr_list = [forward_function(aug)[0] for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output

    def summarize(self, ckp):
        ckp.write_log('# parameters: {:,}'.format(
            sum([p.nelement() for p in self.model.parameters()])
        ))

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

        linear = sum([
            l.weight.nelement() for l in self.model.modules() \
            if isinstance(l, nn.Linear)
        ])

        ckp.write_log(
            '1x1: {:,}\n3x3: {:,}\nOthers: {:,}\nLinear:{:,}\n'.format(
                kernels_1x1, kernels_3x3, kernels_others, linear
            ),
            refresh=True
        )
