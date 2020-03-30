import torch.nn as nn
import numpy as numpy
from collections import OrderedDict, namedtuple
import logging

import distiller
import distiller.utils
from .transformer import Transformer
from .transformer import InitilaizationMode
from .transform_utils import parse_conv_args


class KernelSizeTransformer(Transformer):
    def __init__(self, model, optimizer=None,
                 transform_module=None, transform_kernel=None, transform_init_mode=None, overrides=None):
        super(KernelSizeTransformer, self).__init__(model, optimizer=optimizer,
                                                    transform_module=transform_module,
                                                    transform_kernel=transform_kernel,
                                                    transform_init_mode=transform_init_mode,
                                                    overrides=overrides)

        def k_size_transform_fn(module, name, transform_map, transform_kernel, transform_init_method=InitilaizationMode.NONE):
            # init_method = transform_map.init_method
            if not isinstance(module, nn.Conv2d):
                raise ValueError('Only support nn.conv2d')

            args_conv = parse_conv_args(module)
            args_new = args_conv
            args_new[2] = transform_map.kernel
            if transform_kernel is not None:
                args_new[2] = transform_kernel

            #Calculate padding size for same feature map size output
            pad = int((args_new[2]-1)/2)

            args_new[4] = pad
            if transform_init_method is InitilaizationMode.NONE:
                new_module = nn.Conv2d(*args_new)
            else:
                new_module = nn.Conv2d(*args_new)
            return new_module

        self.default_replacement_fn = k_size_transform_fn
