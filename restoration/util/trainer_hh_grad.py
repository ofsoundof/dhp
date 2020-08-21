import os
from decimal import Decimal
from util import utility
from model_dhp.flops_counter_dhp import set_output_dimension, get_parameters, get_flops
from model.flops_counter import get_model_flops
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from tqdm import tqdm
from loss import similarity_preserving
from IPython import embed


class Trainer(object):
    def __init__(self, args, loader, my_model, my_loss, ckp, writer=None, converging=False, model_teacher=None):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.model_teacher = model_teacher
        self.loss = my_loss
        if args.distillation_final == 'l2' or args.distillation_inter == 'l2':
            self.distill_loss = nn.MSELoss()
        elif args.distillation_final == 'l1' or args.distillation_inter == 'l1':
            self.distill_loss = nn.L1Loss()
        self.writer = writer
        # if not converging:
        self.optimizer = utility.make_optimizer_dhp(args, self.model, ckp, converging=converging)
        self.scheduler = utility.make_scheduler_dhp(args, self.optimizer, int(args.lr_decay_step.split('+')[0]), converging=converging)
        if self.args.model.lower().find('unet') >= 0 or self.args.model.lower().find('dncnn') >= 0:
            self.input_dim = (1, args.input_dim, args.input_dim)
        else:
            self.input_dim = (3, args.input_dim, args.input_dim)
        # embed()
        set_output_dimension(self.model.get_model(), self.input_dim)
        self.flops = get_flops(self.model.get_model())
        self.flops_prune = self.flops  # at initialization, no pruning is conducted.
        self.flops_compression_ratio = self.flops_prune / self.flops
        self.params = get_parameters(self.model.get_model())

        self.params_prune = self.params
        self.params_compression_ratio = self.params_prune / self.params
        self.flops_ratio_log = []
        self.params_ratio_log = []
        self.converging = converging
        self.ckp.write_log('\nThe computation complexity and number of parameters of the current network is as follows.'
                           '\nFlops: {:.4f} [G]\tParams {:.2f} [k]'.format(self.flops / 10. ** 9,
                                                                           self.params / 10. ** 3))
        self.flops_another = get_model_flops(self.model.get_model(), self.input_dim, False)
        self.ckp.write_log('Flops: {:.4f} [G] calculated by the original counter. \nMake sure that the two calculated '
                           'Flops are the same.\n'.format(self.flops_another / 10. ** 9))

        self.error_last = 1e8

    def reset_after_searching(self):
        # During the reloading and testing phase, the searched sparse model is already loaded at initialization.
        # During the training phase, the searched sparse model is just there.
        if not self.converging and not self.args.test_only:
            self.model.get_model().reset_after_searching()
            self.converging = True
            del self.optimizer, self.scheduler
            torch.cuda.empty_cache()
            self.optimizer = utility.make_optimizer_dhp(self.args, self.model, converging=self.converging)
            self.scheduler = utility.make_scheduler_dhp(self.args, self.optimizer, int(self.args.lr_decay_step.split('+')[1]),
                                                        converging=self.converging)

        if self.args.distillation_inter in ['l1', 'l2'] and not self.args.test_only:
            self.model.get_model().add_transform()
            self.model.get_model().cuda()

        if self.args.test_only:
            self.model.get_model().set_channels()

        # print(self.model.get_model())
        print(self.model.get_model(), file=self.ckp.log_file)

        # calculate flops and number of parameters
        self.flops_prune = get_flops(self.model.get_model())
        self.flops_compression_ratio = self.flops_prune / self.flops
        self.params_prune = get_parameters(self.model.get_model())
        self.params_compression_ratio = self.params_prune / self.params

        # reset tensorboardX summary
        if not self.args.test_only and self.args.summary:
            self.writer = SummaryWriter(os.path.join(self.args.dir_save, self.args.save), comment='converging')

        # get the searching epochs
        if os.path.exists(os.path.join(self.ckp.dir, 'epochs.pt')):
            self.epochs_searching = torch.load(os.path.join(self.ckp.dir, 'epochs.pt'))

    def train(self):
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        learning_rate = self.scheduler.get_lr()[0]
        idx_scale = self.args.scale
        if not self.converging:
            stage = 'Searching Stage'
        else:
            stage = 'Finetuning Stage (Searching Epoch {})'.format(self.epochs_searching)
        self.ckp.write_log('\n[Epoch {}]\tLearning rate: {:.2e}\t{}'.format(epoch, Decimal(learning_rate), stage))
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()

        for batch, (lr, hr, _) in enumerate(self.loader_train):
            # if batch <= 1200:
            # if self.args.model == 'EDSR_HH'  and not self.converging:
            #     batch_size = hr.shape[0] // 4
            #     hr = hr[:batch_size]
            #     lr = lr[:batch_size]
            lr, hr = self.prepare([lr, hr])

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(idx_scale, lr)

            if (not self.converging and self.args.distillation_stage == 'c') or \
                    (self.converging and not self.args.distillation_final):
                loss = self.loss(sr, hr)
            else:
                with torch.no_grad():
                    sr_teacher = self.model_teacher(idx_scale, lr)
                if not self.args.distillation_inter:
                    sr = [sr]
                    sr_teacher = [sr_teacher]
                loss, _ = self.loss(sr[0], lr)

                if self.args.distillation_final in ['l1', 'l2']:
                    loss_distill_final = self.distill_loss(sr[0], sr_teacher[0])
                    loss = 0.4 * loss_distill_final + 0.6 * loss
                elif self.args.distillation_inter == 'sp':
                    loss_distill_final = similarity_preserving(sr[0], sr_teacher[0]) * 3000
                    loss = loss_distill_final + loss
                if self.args.distillation_inter in ['l1', 'l2']:
                    loss_distill_inter = 0
                    for p, pt in zip(sr[1], sr_teacher[1]):
                        loss_distill_inter += self.distill_loss(p, pt)
                    loss_distill_inter = loss_distill_inter / len(sr[1]) * self.args.distill_beta
                    loss = loss_distill_inter + loss
                elif self.args.distillation_inter == 'sp':
                    loss_distill_inter = 0
                    for p, pt in zip(sr[1], sr_teacher[1]):
                        loss_distill_inter += similarity_preserving(p, pt)
                    loss_distill_inter = loss_distill_inter / len(sr[1]) * 3000 * self.args.distill_beta
                    loss = loss_distill_inter + loss

            if loss.item() < self.args.skip_threshold * self.error_last:
                # Adam
                loss.backward()
                self.optimizer.step()
                # proximal operator
                if not self.converging:
                    self.model.get_model().proximal_operator(learning_rate)
                    # check the compression ratio
                    if (batch + 1) % self.args.compression_check_frequency == 0 and self.args.use_prox:
                        # set the channels of the potential pruned model
                        self.model.get_model().set_parameters()
                        # update the flops and number of parameters
                        self.flops_prune = get_flops(self.model.get_model())
                        self.flops_compression_ratio = self.flops_prune / self.flops
                        self.params_prune = get_parameters(self.model.get_model())
                        self.params_compression_ratio = self.params_prune / self.params
                        self.flops_ratio_log.append(self.flops_compression_ratio)
                        self.params_ratio_log.append(self.params_compression_ratio)
                        if self.terminate():
                            break
                    if (batch + 1) % 1000 == 0:
                        self.model.get_model().latent_vector_distribution(epoch, batch + 1, self.ckp.dir)
                        self.model.get_model().per_layer_compression_ratio(epoch, batch + 1, self.ckp.dir)

            else:
                print('Skip this batch {}! (Loss: {}) (Threshold: {})'.
                      format(batch + 1, loss.item(), self.args.skip_threshold * self.error_last))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                s = '[{}/{}]\t{}'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch))
                if self.converging or (not self.converging and self.args.distillation_stage == 's'):
                    if self.args.distillation_final or self.args.distillation_inter:
                        s += '\t'
                    if self.args.distillation_final:
                        s += 'DFinal: {:.3f} '.format(loss_distill_final)
                    if self.args.distillation_inter:
                        s += 'DInter: {:.3f}'.format(loss_distill_inter)
                s += '\t{:.3f}+{:.3f}s'.format(timer_model.release(), timer_data.release())
                if hasattr(self, 'flops_compression_ratio') and hasattr(self, 'params_compression_ratio'):
                    s += '\tFlops Ratio: {:.2f}% = {:.4f} G / {:.4f} G' \
                         '\tParams Ratio: {:.2f}% = {:.2f} k / {:.2f} k' \
                        .format(self.flops_compression_ratio * 100,
                                self.flops_prune / 10. ** 9, self.flops / 10. ** 9,
                                self.params_compression_ratio * 100,
                                self.params_prune / 10. ** 3, self.params / 10. ** 3)
                self.ckp.write_log(s)

            timer_data.tic()
            # else:
            #     break
            if not self.converging and epoch == self.args.epochs_grad and batch == 1:
                break

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        # self.error_last = loss
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

                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare([lr, hr])
                    else:
                        lr = self.prepare([lr])[0]

                    torch.cuda.synchronize()
                    timer_test.tic()
                    sr = self.model(idx_scale, lr)
                    torch.cuda.synchronize()
                    timer_test.hold()

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

                mem = torch.cuda.max_memory_allocated()/1024.0**3
                mean_time = timer_test.release() / len(self.loader_test) * 1000

                # Note for testing during traing, the Memory Comsumption may not be right since it is the maximum
                # allocated GPU memory.
                if self.args.model.lower().find('unet') >= 0 or self.args.model.lower().find('dncnn') >= 0:
                    setting = 'Sigma{}'.format(self.args.noise_sigma)
                else:
                    setting = 'x{}'.format(scale)

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} {}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})\t{}\tGPU time: {:.4f} [ms]\tGPU Memory: {:.4f} [GB]'.format(
                        self.args.data_test,
                        setting,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1,
                        self.args.model,
                        mean_time,
                        mem
                    )
                )
                if self.converging:
                    best = self.ckp.log[:self.epochs_searching].max(0)
                    # self.ckp.write_log('\nBest during searching')
                    self.ckp.write_log(
                        '[{} {}]\tPSNR: Best during searching: {:.3f} @epoch {}'.format(
                            self.args.data_test,
                            setting,
                            best[0][idx_scale],
                            best[1][idx_scale] + 1,
                        )
                    )

                with open('./time_memory_psnr.txt', 'a') as f:
                    f.write('Model: {:<20}\tDataset: {:<20}\tGPU time: {:.4f} [ms]\tGPU Memory: {:.4f} [GB]\tPSNR: {:.4f} [dB]\n'.
                            format(self.args.model, self.args.data_test, mean_time, mem, best[0][idx_scale]))

        if not self.args.test_only:
            self.ckp.save(self, epoch, converging=self.converging, is_best=(best[1][0] + 1 == epoch))
        else:
            torch.save(self.model.get_model().state_dict(),
                       os.path.join(self.ckp.dir, '{}_X{}_L{}.pt'.format(self.args.model, self.args.scale[0], self.args.n_resblocks)))

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
            if self.converging:
                return epoch >= self.args.epochs
            else:
                return epoch > self.args.epochs_grad



