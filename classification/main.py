"""
The main function to train a pure network without any modification.
"""
import torch
from util import utility
from data import Data
from model import Model
from loss import Loss
from util.trainer_clean import Trainer
from util.option_basis import args
from tensorboardX import SummaryWriter
import os
from model_dhp.flops_counter_dhp import set_output_dimension, get_parameters, get_flops
from model.in_use.flops_counter import get_model_flops


torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:

    # model
    # if args.decomp_type == 'gsvd' or args.decomp_type == 'svd-mse' or args.comp_rule.find('f-norm') >= 0 \
    #         or args.model.lower().find('prune_resnet56') >= 0:
    #     my_model = Model(args, checkpoint, loader.loader_train)
    # else:
    my_model = Model(args, checkpoint)

    if args.data_train.find('CIFAR') >= 0:
        input_dim = (3, 32, 32)
    elif args.data_train.find('Tiny_ImageNet') >= 0:
        input_dim = (3, 64, 64)
    else:
        input_dim = (3, 224, 224)

    set_output_dimension(my_model.get_model(), input_dim)
    flops = get_flops(my_model.get_model())
    params = get_parameters(my_model.get_model())
    print('\nThe computation complexity and number of parameters of the current network is as follows.'
                       '\nFlops: {:.4f} [G]\tParams {:.2f} [k]'.format(flops / 10. ** 9, params / 10. ** 3))
    flops_another = get_model_flops(my_model.get_model(), input_dim, False)
    print('Flops: {:.4f} [G] calculated by the original counter. \nMake sure that the two calculated '
                       'Flops are the same.\n'.format(flops_another / 10. ** 9))

    # data loader
    loader = Data(args)
    # loss function
    loss = Loss(args, checkpoint)
    # writer
    writer = SummaryWriter(os.path.join(args.dir_save, args.save), comment='optimization') if args.summary else None
    # trainer
    t = Trainer(args, loader, my_model, loss, checkpoint, writer)
    # print('Mem 1 {:2.4f}'.format(torch.cuda.max_memory_allocated()/1024.0**3))
    while not t.terminate():
        # if t.scheduler.last_epoch == 0 and not args.test_only:
        #     t.test()
        # if t.scheduler.last_epoch + 1 == 2:
        t.train()
        t.test()

    checkpoint.done()

