import torch
from util import utility
import data
import model
import loss
from util.option import args
from util.trainer_multi_edsr import Trainer
from util.trainer_finetune import TrainerFT
from model.flops_counter import get_model_activation, get_model_flops

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)
    # embed()
    model = model.Model(args, checkpoint)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    if args.model.lower() == 'finetune':
        t = TrainerFT(args, loader, model, loss, checkpoint)
    else:
        t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()

    input_dim = (3, 512, 512)

    activation, num_conv = get_model_activation(t.model.get_model(), input_dim)
    print('The activation count is {}.\nThe number of conv is {}'.format(activation, num_conv))

    # get the flops
    flops = get_model_flops(model, input_dim)
    print('The FLOPs is {:<2.4f} [M].'.format(flops / 10 ** 6))

    checkpoint.done()

