#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch.nn as nn
import argparse
from enum import Enum
from collections import OrderedDict
from functools import reduce, partial
import logging
import os

import distiller
import distiller.utils
from .quantizer import Quantizer
from .q_utils import *
import distiller.modules

msglogger = logging.getLogger()


class FixedQuantMode(Enum):
    SYMMETRIC = 1
    ASYMMETRIC_UNSIGNED = 2
    ASYMMETRIC_SIGNED = 3

def verify_mode(mode):
    if isinstance(mode, str):
        try:
            return FixedQuantMode[mode]
        except KeyError:
            raise ValueError('Unknown quantization mode string')
    elif isinstance(mode, FixedQuantMode):
        return mode
    else:
        raise TypeError("'mode' argument can be either a string or member of {0}".format(LinearQuantMode.__name__))

def symmetric_fixed_quantization_params(num_bits, saturation_val):
    is_scalar, sat_val = _prep_saturation_val_tensor(saturation_val)

    if any(sat_val < 0):
        raise ValueError('Saturation value must be >= 0')

    # Leave one bit for sign
    n = 2 ** (num_bits - 1) - 1

    # If float values are all 0, we just want the quantized values to be 0 as well. So overriding the saturation
    # value to 'n', so the scale becomes 1
    sat_val[sat_val == 0] = n
    scale = n / sat_val
    dec_width = torch.round(torch.log2(scale))
    frac_width = n - shift_k - 1 #Leave one bit for sign
    zero_point = torch.zeros_like(scale)

    if is_scalar:
        # If input was scalar, return scalars
        return dec_width.item(), frac_width.item(), zero_point.item()

def _get_quant_fixed_params_from_tensor(tensor, num_bits, mode, clip=False, per_channel=False):
    if per_channel and tensor.dim() not in [2, 4]:
        raise ValueError('Per channel quantization possible only with 2D or 4D tensors (linear or conv layer weights)')
    dim = 0 if clip or per_channel else None
    if mode == LinearQuantMode.SYMMETRIC:
        sat_fn = get_tensor_avg_max_abs if clip else get_tensor_max_abs
        sat_val = sat_fn(tensor, dim)
        scale, zp = symmetric_fixed_quantization_params(num_bits, sat_val)
    else:   # Asymmetric mode
        raise RuntimeError('QuantAwareTrainDyamicFixedQuantizer currently does not support running with Asymmetric mode'
        sat_fn = get_tensor_avg_min_max if clip else get_tensor_min_max
        sat_min, sat_max = sat_fn(tensor, dim)
        signed = mode == LinearQuantMode.ASYMMETRIC_SIGNED
        scale, zp = asymmetric_linear_quantization_params(num_bits, sat_min, sat_max, signed=signed)

    if per_channel:
        # Reshape scale and zero_points so they can be broadcast properly with the weight tensor
        dims = [scale.shape[0]] + [1] * (tensor.dim() - 1)
        scale = scale.view(dims)
        zp = zp.view(dims)

    return scale, zp


class QuantAwareTrainDynamicFixedQuantizer(Quantizer):
    def __init__(self, model, optimizer=None, bits_activations=32, bits_weights=32, bits_overrides=None,
                 quantize_bias=True, mode=LinearQuantMode.SYMMETRIC, ema_decay=0.999, per_channel_wts=False,
                 quantize_inputs=True, num_bits_inputs=None):
        super(QuantAwareTrainRangeLinearQuantizer, self).__init__(model, optimizer=optimizer,
                                                                  bits_activations=bits_activations,
                                                                  bits_weights=bits_weights,
                                                                  bits_overrides=bits_overrides,
                                                                  quantize_bias=quantize_bias,
                                                                  train_with_fp_copy=True)

        if isinstance(model, nn.DataParallel) and len(model.device_ids) > 1:
            raise RuntimeError('QuantAwareTrainDyamicFixedQuantizer currently does not support running with '
                               'multiple GPUs')

        mode = verify_mode(mode)

        self.model.quantizer_metadata['params']['mode'] = str(mode).split('.')[1]
        self.model.quantizer_metadata['params']['ema_decay'] = ema_decay
        self.model.quantizer_metadata['params']['per_channel_wts'] = per_channel_wts
        self.model.quantizer_metadata['params']['quantize_inputs'] = quantize_inputs

        # Keeping some parameters for input quantization
        self.quantize_inputs = quantize_inputs
        if num_bits_inputs is not None:
            self.num_bits_inputs = num_bits_inputs
        else:
            self.num_bits_inputs = bits_activations
        self.mode = mode
        self.decay = ema_decay
        self.per_channel_wts = per_channel_wts

        def fixed_quantize_param(param_fp, param_meta):
            m = param_meta.module
            # We don't quantize the learned weights of embedding layers per-channel, because they're used
            # as inputs in subsequent layers and we don't support per-channel activations quantization yet
            perch = not isinstance(m, nn.Embedding) and per_channel_wts and param_fp.dim() in [2, 4]

            with torch.no_grad():
                scale, zero_point = _get_quant_fixed_params_from_tensor(param_fp, param_meta.num_bits, mode,
                                                                  per_channel=perch)
            setattr(m, param_meta.q_attr_name + '_scale', scale)
            setattr(m, param_meta.q_attr_name + '_zero_point', zero_point)
            out = LinearQuantizeSTE.apply(param_fp, scale, zero_point, True, False)
            return out
