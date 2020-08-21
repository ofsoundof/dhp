import torch
from util import utility
import data
from model_dhp import Model
import loss
from util.option import args
from util.trainer_dhp import Trainer
from model_dhp.dhp_base import plot_compression_ratio
from model_dhp.flops_counter_dhp import set_output_dimension, get_parameters, get_flops
from tensorboardX import SummaryWriter
import os

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:

    # ==================================================================================================================
    # Step 1: Initialize the objects.
    # ==================================================================================================================
    info_path = os.path.join(checkpoint.dir, 'epochs_converging.pt')
    if args.load != '' and os.path.exists(info_path):
        # converging stage
        epoch_continue = torch.load(info_path)
    else:
        # searching stage
        epoch_continue = None
    converging = False if epoch_continue is None else True

    loader = data.Data(args)
    model = Model(args, checkpoint)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    writer = SummaryWriter(os.path.join(args.dir_save, args.save), comment='searching') if args.summary else None
    t = Trainer(args, loader, model, loss, checkpoint, writer, converging)

    # ==================================================================================================================
    # Step 2: Searching -> use proximal gradient method to optimize the hypernetwork parameter and
    #          search the potential backbone network configuration
    #          Model: a sparse model
    # ==================================================================================================================
    if not converging and not args.test_only:
        # In the training phase or loading for searching
        while not t.terminate():
            t.train()
            t.test()
            plot_compression_ratio(t.flops_ratio_log, os.path.join(checkpoint.dir, 'flops_compression_ratio_log.png'),
                                   frequency_per_epoch=len(loader.loader_train) // args.compression_check_frequency)
            plot_compression_ratio(t.params_ratio_log, os.path.join(checkpoint.dir, 'params_compression_ratio_log.png'),
                                   frequency_per_epoch=len(loader.loader_train) // args.compression_check_frequency)
        if args.summary:
            t.writer.close()
        # save the compression ratio log and per-layer compression ratio for the latter use.
        torch.save(t.flops_ratio_log, os.path.join(checkpoint.dir, 'flops_compression_ratio_log.pt'))
        torch.save(t.params_ratio_log, os.path.join(checkpoint.dir, 'params_compression_ratio_log.pt'))
        t.model.get_model().per_layer_compression_ratio(0, 0, checkpoint.dir, save_pt=True)

    # ==================================================================================================================
    # Step 3: Pruning -> prune the derived sparse model and prepare the trainer instance for finetuning or testing
    # ==================================================================================================================
    t.reset_after_optimization()
    if args.print_model:
        print(t.model.get_model())
        print(t.model.get_model(), file=checkpoint.log_file)

    # ==================================================================================================================
    # Step 4: Continue the training / Testing -> continue to train the pruned model to have a higher accuracy.
    # ==================================================================================================================
    while not t.terminate():
        t.train()
        t.test()

    set_output_dimension(model.get_model(), t.input_dim)
    flops = get_flops(model.get_model())
    params = get_parameters(model.get_model())
    print('\nThe computation complexity and number of parameters of the current network is as follows.'
          '\nFlops: {:.4f} [G]\tParams {:.2f} [k]\n'.format(flops / 10. ** 9, params / 10. ** 3))

    if args.summary:
        t.writer.close()
    if args.print_model:
        print(t.model.get_model())
        print(t.model.get_model(), file=checkpoint.log_file)
    # for m in t.model.parameters():
    #     print(m.shape)
    checkpoint.done()

