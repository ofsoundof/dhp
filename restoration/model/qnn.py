from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args, parent=False):
    module = import_module('model.' + args.base.lower())
    if args.precision.find('fix') >= 0:
        precision = int(args.precision[3:])
    else:
        precision = 12

    QuantizeParams.bits_w = precision
    QuantizeParams.bits_b = precision
    QuantizeFeature.bits_f = precision

    class QNN(getattr(module, args.base)):
        def __init__(self, args):
            super(QNN, self).__init__(args, conv=quantized_conv)

    return QNN(args)

def quantized_conv(
    in_channels, out_channels, kernel_size,
    stride=1, padding=None, bias=True):

    if padding is None:
        padding = kernel_size // 2

    return QuantizedConv(
        in_channels, out_channels, kernel_size,
        padding=padding, stride=stride, bias=bias
    )

def quantize(params, bits, offset):
    quantized = params.mul(2**(bits - offset - 1))
    quantized.clamp_(-2**(bits - 1), 2**(bits - 1) - 1)
    quantized.round_()
    quantized.div_(2**(bits - offset - 1))

    return quantized

class QuantizeParams(torch.autograd.Function):
    bits_w = 10
    bits_b = 10
    offset_w = -1
    offset_b = 8

    @staticmethod
    def forward(ctx, weight, bias=None):
        ctx.save_for_backward(bias)
        q_weight = quantize(
            weight, QuantizeParams.bits_w, QuantizeParams.offset_w
        )

        if bias is not None:
            q_bias = quantize(
                bias, QuantizeParams.bits_b, QuantizeParams.offset_b
            )
            return q_weight, q_bias
        else:
            return q_weight

    @staticmethod
    def backward(ctx, grad_weight, grad_bias=None):
        bias = ctx.saved_variables
        if bias is not None:
            return grad_weight, grad_bias
        else:
            return grad_weight

class QuantizeFeature(torch.autograd.Function):
    bits_f = 12
    offset_f = 8

    @staticmethod
    def forward(ctx, feature):
        ctx.save_for_backward(feature)
        q_feature = quantize(
            feature, QuantizeFeature.bits_f, QuantizeFeature.offset_f
        )

        return q_feature

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class QuantizedConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size,
        stride=1, padding=0, bias=True):

        super(QuantizedConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        self.kwargs = {
            'stride': (stride, stride),
            'padding': (padding, padding)
        }
        nn.modules.conv._ConvNd.reset_parameters(self)
    
    def __repr__(self):
        s = '{}-bit fixed point convolution({}, {}, kernel_size={}, stride={}, padding={})'.format(
                QuantizeParams.bits_w,
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.kwargs['stride'],
                self.kwargs['padding']
            )

        return s

    def forward(self, x):
        qw, qb = QuantizeParams.apply(self.weight, self.bias)
        output = F.conv2d(x, qw, bias=qb, **self.kwargs)
        output = QuantizeFeature.apply(output)

        return output

