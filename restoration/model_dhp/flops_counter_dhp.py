"""
Reference
https://github.com/sovrasov/flops-counter.pytorch.git
"""
import torch
import torch.nn as nn
import numpy as np
from model_dhp.dhp_base import conv_dhp
# from IPython import embed


def set_output_dimension(model, input_res):
    assert type(input_res) is tuple, 'Please provide the size of the input image.'
    assert len(input_res) >= 3, 'Input image should have 3 dimensions.'
    feat_model = add_feature_dimension(model)
    feat_model.eval().start_dimension_add()
    device = list(feat_model.parameters())[-1].device
    batch = torch.FloatTensor(1, *input_res).to(device)
    _ = feat_model(batch)
    feat_model.stop_dimension_add()


def get_flops(model):
    flops = 0
    for module in model.modules():
        if is_supported_instance(module):
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, conv_dhp)):
                flops += conv_calc_flops(module)
            elif isinstance(module, (nn.ReLU, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6)):
                flops += relu_calc_flops(module)
                # if isinstance(module, nn.ReLU):
                #     print(module)
            elif isinstance(module, (nn.Linear)):
                flops += linear_calc_flops(module)
            elif isinstance(module, (nn.BatchNorm2d)):
                flops += bn_calc_flops(module)
    return flops


def get_parameters(model):
    parameters = 0
    for module in model.modules():
        if is_supported_instance(module):
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.ReLU, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6)):
                for p in module.parameters():
                    parameters += p.nelement()
            elif isinstance(module, nn.Linear):
                in_features = module.in_features_remain if hasattr(module, 'in_features_remain') else module.in_features
                out_features = module.out_features_remain if hasattr(module, 'out_features_remain') else module.out_features
                parameters += in_features * out_features
                if module.bias is not None:
                    parameters += module.out_features
            elif isinstance(module, (conv_dhp)):
                in_channels = module.in_channels_remain if hasattr(module, 'in_channels_remain') else module.in_channels
                out_channels = module.out_channels_remain if hasattr(module, 'out_channels_remain') else module.out_channels
                groups = module.groups_remain if hasattr(module, 'groups_remain') else module.groups
                parameters += in_channels // groups * out_channels * module.kernel_size ** 2
                if module.bias is not None:
                    parameters += out_channels
            elif isinstance(module, nn.BatchNorm2d):
                if module.affine:
                    num_features = module.num_features_remain if hasattr(module, 'num_features_remain') else module.num_features
                    parameters += num_features * 2
    return parameters


def add_feature_dimension(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_dimension_add = start_dimension_add.__get__(net_main_module)
    net_main_module.stop_dimension_add = stop_dimension_add.__get__(net_main_module)

    return net_main_module


def start_dimension_add(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Activates the computation of mean flops consumption per image.
    Call it before you run the network.

    """
    self.apply(add_feat_dim_hook_function)


def stop_dimension_add(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Stops computing the mean flops consumption per image.
    Call whenever you want to pause the computation.

    """
    self.apply(remove_feat_dim_hook_function)


def add_feat_dim_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            return

        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, conv_dhp)):
            handle = module.register_forward_hook(conv_feat_dim_hook)
        elif isinstance(module, (nn.ReLU, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6)):
            handle = module.register_forward_hook(relu_feat_dim_hook)
        elif isinstance(module, nn.Linear):
            handle = module.register_forward_hook(linear_feat_dim_hook)
        elif isinstance(module, nn.BatchNorm2d):
            handle = module.register_forward_hook(bn_feat_dim_hook)
        else:
            raise NotImplementedError('FLOPs calculation is not implemented for class {}'.format(module.__class__.__name__))
        module.__flops_handle__ = handle


def remove_feat_dim_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            module.__flops_handle__.remove()
            del module.__flops_handle__


# ---- Internal functions
def is_supported_instance(module):
    if isinstance(module,
                  (
                          conv_dhp,
                          nn.Conv2d, nn.ConvTranspose2d,
                          nn.BatchNorm2d,
                          nn.Linear,
                          nn.ReLU, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6,
                  )):
        if hasattr(module, '__exclude_complexity__'):
            return False
        else:
            return True

    return False


def conv_feat_dim_hook(module, input, output):
    module.__output_dims__ = output.shape[2:]


def conv_calc_flops(self):
    # Do not count bias addition
    batch_size = 1
    output_dims = np.prod(self.__output_dims__)

    kernel_dims = np.prod(self.kernel_size) if isinstance(self.kernel_size, tuple) else self.kernel_size ** 2
    in_channels = self.in_channels_remain if hasattr(self, 'in_channels_remain') else self.in_channels
    out_channels = self.out_channels_remain if hasattr(self, 'out_channels_remain') else self.out_channels
    groups = self.groups_remain if hasattr(self, 'groups_remain') else self.groups
    # groups = self.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = kernel_dims * in_channels * filters_per_channel

    active_elements_count = batch_size * output_dims

    overall_conv_flops = conv_per_position_flops * active_elements_count

    return int(overall_conv_flops)


def relu_feat_dim_hook(module, input, output):
    s = output.shape
    module.__output_dims__ = s[2:]
    module.__output_channel__ = s[1]


def relu_calc_flops(self):
    batch = 1
    channels = self.channels if hasattr(self, 'channels') else self.__output_channel__
    active_elements_count = batch * np.prod(self.__output_dims__) * channels
    # print(active_elements_count, id(self))
    # print(self)
    return int(active_elements_count)


def linear_feat_dim_hook(module, input, output):
    if len(output.shape[2:]) == 2:
        module.__additional_dims__ = 1
    else:
        module.__additional_dims__ = output.shape[1:-1]


def linear_calc_flops(self):
    # Do not count bias addition
    batch_size = 1
    in_features = self.in_features_remain if hasattr(self, 'in_features_remain') else self.in_features
    out_features = self.out_features_remain if hasattr(self, 'out_features_remain') else self.out_features
    linear_flops = batch_size * np.prod(self.__additional_dims__) * in_features * out_features
    return int(linear_flops)


def bn_feat_dim_hook(module, input, output):
    module.__output_dims__ = output.shape[2:]


def bn_calc_flops(self):
    # Do not count bias addition
    batch = 1
    output_dims = np.prod(self.__output_dims__)
    channels = self.num_features_remain if hasattr(self, 'num_features_remain') else self.num_features
    batch_flops = batch * channels * output_dims
    if self.affine:
        batch_flops *= 2
    return int(batch_flops)

