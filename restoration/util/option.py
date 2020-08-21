import argparse
from util import template

parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='/scratch_net/ofsoundof/yawli/Datasets',
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='DIV2K',
                    help='test dataset name')
parser.add_argument('--benchmark_noise', action='store_true',
                    help='use noisy benchmark sets')
parser.add_argument('--n_train', type=int, default=800,
                    help='number of training set')
parser.add_argument('--n_val', type=int, default=100,
                    help='number of validation set')
parser.add_argument('--offset_val', type=int, default=800,
                    help='validation index offest')
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--scale', default='4',
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=192,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--noise', type=str, default='.',
                    help='Gaussian noise std.')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')

# Model specifications
parser.add_argument('--model', default='EDSR',
                    help='model name')
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pretrain', type=str, default='',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=5,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# CARN: convolutional anchored regression networks
parser.add_argument('--res_act', default='SIGMOID',
                    help='activation function is res block')
parser.add_argument('--reg_anchor', type=int, default=16,
                    help='number of anchors')
parser.add_argument('--reg_out', type=int, default=16,
                    help='number of channels in the regression block')
parser.add_argument('--submodel', default='carn',
                    help='submodel name')
parser.add_argument('--norm_type', type=str, default='groupnorm',
                    help='Normalization type')
parser.add_argument('--n_groups', type=int, default=4,
                    help='number of groups in group normalization')
# 3D appearance super-resolution
parser.add_argument('--n_resblocks_ft', type=int, default=2,
                    help='number of resblocks used for finetuning')
parser.add_argument('--model_one', default='one',
                    help='used to split the dataset')
parser.add_argument('--subset', default='.',
                    help='extract a subset of the whole dataset')
parser.add_argument('--normal_lr', default='hr',
                    help='use hr or lr normal map')
parser.add_argument('--input_res', default='lr',
                    help='use hr or lr input')
parser.add_argument('--n_resunits', type=int, default=6,
                    help='number of resunits used for level one residual')
# Factor
parser.add_argument('--sic_layer', type=int, default=2,
                    help='number of SIC layers for Factorized convolutional neural networks.')
# Group
parser.add_argument('--group_size', type=int, default=16,
                    help='group size for the network of filter group approximation, ECCV 2018 paper.')
# Basis learning parameters
parser.add_argument('--bn_every', action='store_true',
                    help='Used by srresnet_basis and edsr_basis')
parser.add_argument('--basis_size', type=int, default=64,
                    help='basis size')
parser.add_argument('--n_basis', type=int, default=16,
                    help='number of basis')
parser.add_argument('--share_basis', action='store_true',
                    help='whether to share the basis set for the two convs in the Residual Block')

parser.add_argument('--pre_train_optim', type=str, default='.',
                    help='pre-trained weights directory')
parser.add_argument('--loss_norm', action='store_true',
                    help='whether to use default loss_norm')
# Clusternet
parser.add_argument('--multi', type=str, default='full-256',
                    help='multi clustering')
parser.add_argument('--n_init', type=int, default=1,
                    help='number of differnt k-means initialization')
parser.add_argument('--max_iter', type=int, default=4500,
                    help='maximum iterations for kernel clustering')
parser.add_argument('--symmetry', type=str, default='i',
                    help='clustering algorithm')
parser.add_argument('--init_seeds', type=str, default='random',
                    help='kmeans initialization method')
parser.add_argument('--scale_type', type=str, default='kernel_norm_train',
                    help='scale parameter configurations')
parser.add_argument('--n_bits', type=int, default=16,
                    help='number of bits for scale parameters')
parser.add_argument('--pretrain_cluster', type=str, default='',
                    help='pretrained model for clustering')


# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=2000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=200,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e6',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--dir_save', default='/scratch_net/ofsoundof/yawli/dhp_sr/tmp',
                    help='the directory used to save')
parser.add_argument('--save', type=str, default='test',
                    help='file name to save')
parser.add_argument('--load', type=str, default='',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=-1,
                    help='resume from specific checkpoint')
parser.add_argument('--print_model', action='store_true',
                    help='print model')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')

parser.add_argument('--act_res', type=str, default='Yes', choices=['Yes', 'No'],
                    help='activation in the resblock')
parser.add_argument('--expansion', type=int, default=0,
                    help='expansion')
parser.add_argument('--expansion_batchnorm', type=str, default='Yes', choices=['Yes', 'No'],
                    help='batchnorm in expansion')
parser.add_argument('--expansion_act', type=str, default='Yes', choices=['Yes', 'No'],
                    help='act in expansion')

# options for residual group and feature channel reduction
parser.add_argument('--n_resgroups', type=int, default=5,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')
# options for test
parser.add_argument('--testpath', type=str, default='../test/DIV2K_val_LR_our',
                    help='dataset directory for testing')
parser.add_argument('--testset', type=str, default='Set5',
                    help='dataset name for testing')


# differentiable pruning via hypernetworks
parser.add_argument('--prune_threshold', type=float, default=5e-3,
                    help='prune threshold')
parser.add_argument('--regularization_factor', type=float, default=1e-2,
                    help='pruning regularization factor.')
parser.add_argument('--ratio', type=float, default=0.2,
                    help='compression ratio')
parser.add_argument('--stop_limit', type=float, default=0.05,
                    help='the stop limit of the binary searching method')
parser.add_argument('--embedding_dim', type=float, default=8,
                    help='the dimension of per-location element in the latent matrix')
parser.add_argument('--finetune', action='store_true',
                    help='used to determine whether to load the model for the searching stage or the finetuning stage.')
parser.add_argument('--compression_check_frequency', type=int, default=100,
                    help='check the compression ratio per N batches')
parser.add_argument('--summary', action='store_true',
                    help='add tensorboardX summary to monitor the weights and the gradients')
parser.add_argument('--prune_upsampler', action='store_true',
                    help='prune the upsampler of SRResNet and EDSR.')
parser.add_argument('--input_dim', type=int, default=512,
                    help='the input dimension to calculate the computational complexity FLOPs.')
parser.add_argument('--lr_decay_step', type=str, default='10+200',
                    help='learning rate decay step for differentiable pruning via hypernetworks.')
parser.add_argument('--distillation_final', type=str, default='no', choices=['no', 'l2', 'l1', 'sp'],
                    help='how to apply distillation to the feature map of the final layer')
parser.add_argument('--distillation_inter', type=str, default='no', choices=['no', 'l2', 'l1', 'sp'],
                    help='how to apply distillation to the feature map of the intermediate layer')
parser.add_argument('--distillation_stage', type=str, default='c', choices=['s', 'c'],
                    help='the stage where the distillation is applied:'
                         's: the distillation is added from the searching stage;'
                         'c: the distillation is added from the converging stage.')
parser.add_argument('--distill_beta', type=int, default=1,
                    help='The beta used for the intermediate layers of distillation')
parser.add_argument('--width_mult', type=float, default=1.0,
                    help='The width multiplier')
parser.add_argument('--teacher', type=str, default='',
                    help='pretrained teacher model used for distillation')
parser.add_argument('--grad_prune', action='store_true',
                    help='Prune the latent vector according their gradients.')
parser.add_argument('--grad_normalize', default='No', choices=['Yes', 'No'],
                    help='Normalize the gradients of the latnet vectors.')
parser.add_argument('--epochs_grad', type=int, default=10,
                    help='The number of epochs during pruning stage.')
parser.add_argument('--use_prox', type=str, default='No', choices=['Yes', 'No'],
                    help='Whether to use the proximal operator to sparsify the latent vector during warm-up training.')
parser.add_argument('--remain_percentage', type=float, default=-1,
                    help='this indicates the minimum percentage of remaining channels after the compression procedure.'
                         'If it is -1, this option is not valid.')

# denoising
parser.add_argument('--noise_sigma', type=int, default=50,
                    help='Gaussian noise std.')
parser.add_argument('--quality', type=int, default=10,
                    help='quality factor of jpeg')
parser.add_argument('--bn', default=False,
                    help='use batch normalization in DNCNN.')
parser.add_argument('--m_blocks', type=int, default=4,
                    help='number of blocks in DNCNN.')

args = parser.parse_args()
template.set_template(args)

args.scale = list(map(lambda x: int(x), args.scale.split('+')))

if args.epochs == 0:
    args.epochs = 1e8

if args.distillation_final == 'no':
    args.distillation_final = ''
if args.distillation_inter == 'no':
    args.distillation_inter = ''

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False



