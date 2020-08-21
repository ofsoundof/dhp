import os
from decimal import Decimal
from util import utility
import torch
import torch.nn as nn
from tqdm import tqdm
from model_dhp.flops_counter_dhp import set_output_dimension, get_parameters, get_flops
from model.flops_counter import get_model_flops

class Trainer(object):
    def __init__(self, args, loader, my_model, my_loss, ckp, teacher=None):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.teacher = teacher
        self.loss = my_loss
        if args.distillation_final == 'l2' or args.distillation_inter == 'l2':
            self.distill_loss = nn.MSELoss()
        elif args.distillation_final == 'l1' or args.distillation_inter == 'l1':
            self.distill_loss = nn.L1Loss()
        self.optimizer = utility.make_optimizer_dhp(args, self.model, ckp=ckp)
        self.scheduler = utility.make_scheduler_dhp(args, self.optimizer, args.lr_decay)

        if self.args.model.lower().find('unet') >= 0 or self.args.model.lower().find('dncnn') >= 0:
            self.input_dim = (1, args.input_dim, args.input_dim)
        else:
            self.input_dim = (3, args.input_dim, args.input_dim)
        set_output_dimension(self.model.get_model(), self.input_dim)
        self.flops = get_flops(self.model.get_model())
        self.params = get_parameters(self.model.get_model())
        self.ckp.write_log('\nThe computation complexity and number of parameters of the current network is as follows.'
                           '\nFlops: {:.4f} [G]\tParams {:.2f} [k]'.format(self.flops / 10. ** 9,
                                                                           self.params / 10. ** 3))
        self.flops_another = get_model_flops(self.model.get_model(), self.input_dim, False)
        self.ckp.write_log('Flops: {:.4f} [G] calculated by the original counter. \nMake sure that the two calculated '
                           'Flops are the same.\n'.format(self.flops_another / 10. ** 9))

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]
        idx_scale = self.args.scale

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, _) in enumerate(self.loader_train):
            # if batch <= 10:
            lr, hr = self.prepare([lr, hr])
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(idx_scale, lr)
            loss = self.loss(sr, hr)

            if self.args.distillation_final:
                with torch.no_grad():
                    sr_teacher = self.teacher(idx_scale, lr)
                loss_distill = self.distill_loss(sr, sr_teacher)
                loss = 0.4 * loss_distill + 0.6 * loss

            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                s = '[{}/{}]\t{}'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch))
                if self.args.teacher:
                    s += '\tDFinal: {:.3f}'.format(loss_distill)
                s += '\t{:.3f}+{:.3f}s'.format(
                    timer_model.release(),
                    timer_data.release())
                self.ckp.write_log(s)

            timer_data.tic()
            # else:
            #     break

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.scheduler.step()

    def test(self):
        epoch = self.scheduler.last_epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lr, hr, filename) in enumerate(tqdm_test):

                    # from IPython import embed; embed();
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare([lr, hr])
                    else:
                        lr = self.prepare([lr])[0]

                    sr = self.model(idx_scale, lr)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            div2k=self.args.data_test == 'DIV2K'
                        )
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale)

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))
        else:
            torch.save(self.model.get_model().state_dict(),
                       os.path.join(self.ckp.dir,
                                    '{}_X{}_L{}.pt'.format(self.args.model, self.args.scale[0], self.args.n_resblocks)))

    def prepare(self, l):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs

