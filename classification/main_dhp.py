"""
This is the main function used for differentiable pruning via hypernetworks
"""
import torch
from util import utility
import os
from data import Data
from model_dhp import Model
from loss import Loss
from util.trainer_dhp import Trainer
from util.option_dhp import args
from tensorboardX import SummaryWriter
from model_dhp.dhp_base import plot_compression_ratio
from model_dhp.flops_counter_dhp import set_output_dimension, get_parameters, get_flops


torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    """
    Four phases.
    Phase 1: training from scratch.                   load = '', pretrain = '' 
             -> not loading the model
             -> not loading the optimizer
    Phase 2: testing phase; test_only.                load = '', pretrain = '*/model/model_latest.pt' or '*/model/model_merge_latest.pt'
             -> loading the pretrained model
             -> not loading the optimizer
    Phase 3: loading models for PG optimization.      load = '*/' -- a directory, pretrain = '', epoch_continue = None
             -> loading from model_latest.pt
             -> loading optimizer.pt
    Phase 4: loading models to continue the training. load = '*/' -- a directory, pretrain = '', epoch_continue = a number
             -> loading from model_continue.pt
             -> loading optimizer_converging.pt
    During the loading phase (3 & 4), args.load is set to a directory. The loaded model is determined by the 'stage' of 
            the algorithm. 
    Thus, need to first determine the stage of the algorithm. 
    Then decide whether to load model_latest.pt or model_continue_latest.pt

    The algorithm has two stages, i.e, proximal gradient (PG) optimization (searching stage) and continue-training 
            (converging stage). 
    The stage is determined by epoch_continue. The variable epoch_continue denotes the epoch where the PG optimization
            finishes and the training continues until the convergence of the network.
    i.  epoch_continue = None -> PG optimzation stage (searching stage)
    ii. epoch_continue = a number -> continue-training stage (converging stage)
    
    Initial epoch_continue:
        Phase 1, 2, &3 -> epoch_continue = None, converging = False
        PHase 4 -> epoch_continue = a number, converging = True
    """

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
    # Judge which stage the algorithm is in, namely the searching stage or the converging stage.
    converging = False if epoch_continue is None else True

    loader = Data(args)
    loss = Loss(args, checkpoint)
    network_model = Model(args, checkpoint, converging=converging)
    writer = SummaryWriter(checkpoint.dir, comment='searching') if args.summary else None
    t = Trainer(args, loader, network_model, loss, checkpoint, writer, converging)

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
    t.reset_after_searching()
    if args.print_model:
        print(t.model.get_model())
        print(t.model.get_model(), file=checkpoint.log_file)
    # for k, v in t.model.state_dict(keep_vars=True).items():
    #     print(list(v.shape), k)

    # ==================================================================================================================
    # Step 4: Continue the training / Testing -> continue to train the pruned model to have a higher accuracy.
    # ==================================================================================================================

    while not t.terminate():
        t.train()
        t.test()

    set_output_dimension(network_model.get_model(), t.input_dim)
    flops = get_flops(network_model.get_model())
    params = get_parameters(network_model.get_model())
    print('\nThe computation complexity and number of parameters of the current network is as follows.'
          '\nFlops: {:.4f} [G]\tParams {:.2f} [k]\n'.format(flops / 10. ** 9, params / 10. ** 3))

    if args.summary:
        t.writer.close()
    if args.print_model:
        print(t.model.get_model())
        print(t.model.get_model(), file=checkpoint.log_file)
    checkpoint.done()

