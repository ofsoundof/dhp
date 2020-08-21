from util.option import parser
from util import template


# differentiable pruning via hypernetworks
parser.add_argument('--prune_threshold', type=float, default=5e-3,
                    help='the threshold used to mask out or nullifying the small elements.')
parser.add_argument('--regularization_factor', type=float, default=1e-4,
                    help='the sparsity regularization factor.')
parser.add_argument('--stop_limit', type=float, default=0.05,
                    help='the stop limit of the binary searching method')
parser.add_argument('--prune_same_channels', type=str, default='Yes', choices=['Yes', 'No'],
                    help='whether to prune the same channels for the blocks in the same stage of ResNet')
parser.add_argument('--embedding_dim', type=int, default=8,
                    help='the dimension of per-location element in the latent matrix')
# parser.add_argument('--finetune', action='store_true',
#                     help='used to determine whether to load the model for the searching stage or the finetuning stage.')
parser.add_argument('--compression_check_frequency', type=int, default=100,
                    help='check the compression ratio per N batches')
parser.add_argument('--mc', type=int, default=4,
                    help='minimum number of remaining channels')
parser.add_argument('--remain_percentage', type=float, default=-1,
                    help='this indicates the minimum percentage of remaining channels after the compression procedure.'
                         'If it is -1, this option is not valid.')
parser.add_argument('--linear_percentage', type=float, default=0.4,
                    help='')
parser.add_argument('--teacher', type=str, default='',
                    help='pretrained teacher model used for distillation')
parser.add_argument('--distillation_final', type=str, default='no', choices=['no', 'kd', 'sp'],
                    help='how to apply distillation to the feature map of the final layer')
parser.add_argument('--distillation_inter', type=str, default='no', choices=['no', 'kd', 'sp'],
                    help='how to apply distillation to the feature map of the intermediate layer')
parser.add_argument('--distillation_stage', type=str, default='c', choices=['s', 'c'],
                    help='the stage where the distillation is applied:'
                         's: the distillation is added from the searching stage;'
                         'c: the distillation is added from the converging stage.')
parser.add_argument('--distill_beta', type=int, default=1,
                    help='The beta used for the intermediate layers of distillation')
parser.add_argument('--temperature', type=float, default=4,
                    help='Temperature of distillation.')
parser.add_argument('--prune_classifier', action='store_true',
                    help='Whether to prune the classifier of MobileNetV2')
parser.add_argument('--conv3_expand', action='store_true',
                    help='Whether to expand the output channels of the third convolution of CIFAR-Quick')

parser.add_argument('--sparsity_regularizer', type=str, default='l1', choices=['l1', 'l2'],
                    help='the types of sparsity regularizer applied.')

# Pruning the latent vector according their gradients at network initialization or after training the hyper network for
# a small number of epochs (contains an initial warm-up training).
# Several decisions need to be made for this method.
# 1. Need to decide how many warm-up epochs should be used.
# 2. In the warm-up training, need to decide whether to use the proximal operator.
parser.add_argument('--grad_prune', action='store_true',
                    help='Prune the latent vector according their gradients.')
parser.add_argument('--grad_normalize', default='No', choices=['Yes', 'No'],
                    help='Normalize the gradients of the latnet vectors.')
parser.add_argument('--epochs_grad', type=int, default=10,
                    help='The number of epochs during pruning stage.')
parser.add_argument('--use_prox', type=str, default='No', choices=['Yes', 'No'],
                    help='Whether to use the proximal operator to sparsify the latent vector during warm-up training.')
parser.add_argument('--epochs_search', type=int, default=-1,
                    help='Searching epoch.')

args = parser.parse_args()
template.set_template(args)

if args.distillation_final == 'no':
    args.distillation_final = ''
if args.distillation_inter == 'no':
    args.distillation_inter = ''

args.grad_normalize = args.grad_normalize == 'Yes'
args.use_prox = args.use_prox == 'Yes'

