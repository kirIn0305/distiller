import torch.nn as nn

def parse_conv_args(module):
    if not isinstance(module, nn.Conv2d):
        raise ValueError(f'This module {module} is not nn.Conv2d')
    in_channels = module.in_channels
    out_channels = module.out_channels
    kernel_size = module.kernel_size
    stride = module.stride
    padding = module.padding
    dilation = module.dilation
    groups = module.groups
    bias = module.bias
    padding_mode = module.padding_mode
    args_conv = [in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode]
    return args_conv