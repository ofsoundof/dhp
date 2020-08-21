import os
from decimal import Decimal
import torch.nn as nn
from util import utility

import torch
from tqdm import tqdm

class Trainer(object):
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):
            lr, hr = self.prepare([lr, hr])
            timer_data.hold()
            timer_model.tic()
            self.optimizer.zero_grad()
            from IPython import embed; embed(); exit()
            sr, weights, biases = self.model(idx_scale, lr)
            #from IPython import embed; embed(); exit()
            # loss_weight_norm = 0.05 * (torch.sum(weights[1] ** 2)/256/256/9 + torch.sum(weights[2] ** 2)/256/27/9)
            loss_weight_norm = 0.05 * (torch.sum(self.model.model.basis ** 2)/64/16/9)
            loss_weight = self.loss_weight(weights, 'L2')
            loss = 5 * loss_weight + 20 * loss_weight_norm + self.loss(sr, hr)
            #self.loss.log[-1, -1] = self.loss.log[-1, -1] + loss_weight.item() #+ loss_weight_norm

            if loss.item() < self.args.skip_threshold * self.error_last:
                # from IPython import embed; embed();
                loss.backward()
                # from IPython import embed; embed();
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {}) (Threshold: {})'.format(
                    batch + 1, loss.item(), self.args.skip_threshold * self.error_last
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                print('Loss: {:<2.8f}/ Loss_weight: {:<2.8f}/ Weight_norm: {:<2.8f}'.format(loss.item(), loss_weight.item(), loss_weight_norm.item()))
                print(idx_scale)
                self.ckp.write_log('[{}/{}]\t{}\t{:.3f}+{:.3f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        # self.error_last = loss

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lr, hr, filename, _) in enumerate(tqdm_test):

                    # from IPython import embed; embed();
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare([lr, hr])
                    else:
                        lr = self.prepare([lr])[0]

                    sr = self.model(idx_scale, lr)
                    # from IPython import embed; embed();
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale)
                # from IPython import embed; embed();
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

    def prepare(self, l, volatile=False):
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

    def loss_weight(self, weights_train, para_loss_type):
        para = torch.load('/scratch_net/ofsoundof/yawli/conadp/softplus_SRBASIS_X3_L20B16P192F256/model/model_latest.pt')
        keys = [k for k, _ in para.items()]
        weights_tar = [para[keys[2]], para[keys[5]], para[keys[8]]]
        loss = 0
        loss_fun = self.loss_type(para_loss_type)
        for i in range(3):
            loss += loss_fun(weights_tar[i], weights_train[i])
            # from IPython import embed; embed();
        return loss

    def loss_type(self, loss_para_type):
        if loss_para_type == 'L1':
            loss_fun = nn.L1Loss()
        elif loss_para_type == 'L2':
            loss_fun = nn.MSELoss()
        else:
            raise NotImplementedError
        return loss_fun
