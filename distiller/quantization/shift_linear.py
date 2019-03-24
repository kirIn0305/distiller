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
from .range_linear import LinearQuantMode
from .range_linear import verify_mode
from .range_linear import update_ema
from .range_linear import inputs_quantize_wrapped_forward


class ShiftQuantMode(Enum):
    CEIL = 1
    FLOOR = 2
    ROUND = 3

def _bit_shift_quantization_params(scale, mode):
    log_scale = torch.log2(scale)
    
    if mode == ShiftQuantMode.CEIL:
        shift_k = torch.ceil(log_scale)
    elif mode == ShiftQuantMode.FLOOR:
        shift_k = torch.floor(log_scale)
    elif mode == ShiftQuantMode.ROUND:
        shift_k = torch.round(log_scale)
    else:
        raise ValueError('_bit_shift_quantization_params not support mode {0}'.format(mode))

    scale_shift = torch.pow(2, shift_k)
    return scale_shift

def _get_quant_shift_params_from_tensor(tensor, num_bits, linear_mode, shift_mode, clip=False, per_channel=False):
    if per_channel and tensor.dim() not in [2, 4]:
        raise ValueError('Per channel quantization possible only with 2D or 4D tensors (linear or conv layer weights)')
    dim = 0 if clip or per_channel else None
    if linear_mode == LinearQuantMode.SYMMETRIC:
        sat_fn = get_tensor_avg_max_abs if clip else get_tensor_max_abs
        sat_val = sat_fn(tensor, dim)
        scale_f, zp = symmetric_linear_quantization_params(num_bits, sat_val)
    else:   # Asymmetric mode
        sat_fn = get_tensor_avg_min_max if clip else get_tensor_min_max
        sat_min, sat_max = sat_fn(tensor, dim)
        signed = linear_mode == LinearQuantMode.ASYMMETRIC_SIGNED
        scale_f, zp = asymmetric_linear_quantization_params(num_bits, sat_min, sat_max, signed=signed)

    scale_shift = _bit_shift_quantization_params(scale_f, shift_mode)

    if per_channel:
        # Reshape scale and zero_points so they can be broadcast properly with the weight tensor
        dims = [scale_shift.shape[0]] + [1] * (tensor.dim() - 1)
        scale_shift = scale_shift.view(dims)
        zp = zp.view(dims)

    return scale_shift, zp


class FakeShiftQuantization(nn.Module):
    def __init__(self, num_bits=8, linear_mode=LinearQuantMode.SYMMETRIC, shift_mode=ShiftQuantMode.CEIL, ema_decay=0.999, dequantize=True, inplace=False):
        super(FakeShiftQuantization, self).__init__()

        self.num_bits = num_bits
        self.linear_mode = linear_mode
        self.shift_mode = shift_mode
        self.dequantize = dequantize
        self.inplace = inplace

        # We perform bias correction on the EMA, so we keep both unbiased and biased values and the iterations count
        # For a simple discussion of this see here:
        # https://www.coursera.org/lecture/deep-neural-network/bias-correction-in-exponentially-weighted-averages-XjuhD
        self.register_buffer('ema_decay', torch.tensor(ema_decay))
        self.register_buffer('tracked_min_biased', torch.zeros(1))
        self.register_buffer('tracked_min', torch.zeros(1))
        self.register_buffer('tracked_max_biased', torch.zeros(1))
        self.register_buffer('tracked_max', torch.zeros(1))
        self.register_buffer('iter_count', torch.zeros(1))
        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('zero_point', torch.zeros(1))

    def forward(self, input):
        # We update the tracked stats only in training
        #
        # Due to the way DataParallel works, we perform all updates in-place so the "main" device retains
        # its updates. (see https://pytorch.org/docs/stable/nn.html#dataparallel)
        # However, as it is now, the in-place update of iter_count causes an error when doing
        # back-prop with multiple GPUs, claiming a variable required for gradient calculation has been modified
        # in-place. Not clear why, since it's not used in any calculations that keep a gradient.
        # It works fine with a single GPU. TODO: Debug...
        if self.training:
            with torch.no_grad():
                current_min, current_max = get_tensor_min_max(input)
            self.iter_count += 1
            self.tracked_min_biased.data, self.tracked_min.data = update_ema(self.tracked_min_biased.data,
                                                                             current_min, self.ema_decay,
                                                                             self.iter_count)
            self.tracked_max_biased.data, self.tracked_max.data = update_ema(self.tracked_max_biased.data,
                                                                             current_max, self.ema_decay,
                                                                             self.iter_count)

        if self.linear_mode == LinearQuantMode.SYMMETRIC:
            max_abs = max(abs(self.tracked_min), abs(self.tracked_max))
            actual_min, actual_max = -max_abs, max_abs
            if self.training:
                self.scale.data, self.zero_point.data = symmetric_linear_quantization_params(self.num_bits, max_abs)
        else:
            actual_min, actual_max = self.tracked_min, self.tracked_max
            signed = self.linear_mode == LinearQuantMode.ASYMMETRIC_SIGNED
            if self.training:
                self.scale.data, self.zero_point.data = asymmetric_linear_quantization_params(self.num_bits,
                                                                                              self.tracked_min,
                                                                                              self.tracked_max,
                                                                                              signed=signed)

        self.scale.data = _bit_shift_quantization_params(self.scale.data, self.shift_mode)


        input = clamp(input, actual_min.item(), actual_max.item(), False)
        input = LinearQuantizeSTE.apply(input, self.scale, self.zero_point, self.dequantize, False)

        return input

    def extra_repr(self):
        linear_mode_str = str(self.linear_mode).split('.')[1]
        return 'linear_mode={0}, num_bits={1}, ema_decay={2:.4f})'.format(linear_mode_str, self.num_bits, self.ema_decay)


class FakeShiftQuantizationWrapper(nn.Module):
    def __init__(self, wrapped_module, num_bits, linear_mode, shift_mode, ema_decay):
        super(FakeShiftQuantizationWrapper, self).__init__()
        self.wrapped_module = wrapped_module
        self.fake_q = FakeShiftQuantization(num_bits, linear_mode, shift_mode, ema_decay, dequantize=True,
                                             inplace=getattr(wrapped_module, 'inplace', False))

    def forward(self, *input):
        res = self.wrapped_module(*input)
        res = self.fake_q(res)
        return res

class QuantAwareTrainShiftLinearQuantizer(Quantizer):
    def __init__(self, model, optimizer=None, bits_activations=32, bits_weights=32, bits_overrides=None,
                 quantize_bias=True, linear_mode=LinearQuantMode.SYMMETRIC, shift_mode=ShiftQuantMode.CEIL, ema_decay=0.999, per_channel_wts=False,
                 quantize_inputs=True, num_bits_inputs=None):
        super(QuantAwareTrainShiftLinearQuantizer, self).__init__(model, optimizer=optimizer,
                                                                  bits_activations=bits_activations,
                                                                  bits_weights=bits_weights,
                                                                  bits_overrides=bits_overrides,
                                                                  quantize_bias=quantize_bias,
                                                                  train_with_fp_copy=True)

        if isinstance(model, nn.DataParallel) and len(model.device_ids) > 1:
            raise RuntimeError('QuantAwareTrainShiftLinearQuantizer currently does not support running with '
                               'multiple GPUs')

        linear_mode = verify_mode(linear_mode)

        self.model.quantizer_metadata['params']['linear_mode'] = str(linear_mode).split('.')[1]
        self.model.quantizer_metadata['params']['shift_mode'] = str(shift_mode).split('.')[1]
        self.model.quantizer_metadata['params']['ema_decay'] = ema_decay
        self.model.quantizer_metadata['params']['per_channel_wts'] = per_channel_wts
        self.model.quantizer_metadata['params']['quantize_inputs'] = quantize_inputs

        # Keeping some parameters for input quantization
        self.quantize_inputs = quantize_inputs
        if num_bits_inputs is not None:
            self.num_bits_inputs = num_bits_inputs
        else:
            self.num_bits_inputs = bits_activations
        self.linear_mode = linear_mode
        self.shift_mode = shift_mode
        self.decay = ema_decay
        self.per_channel_wts = per_channel_wts

        def shift_quantize_param(param_fp, param_meta):
            m = param_meta.module
            # We don't quantize the learned weights of embedding layers per-channel, because they're used
            # as inputs in subsequent layers and we don't support per-channel activations quantization yet
            perch = not isinstance(m, nn.Embedding) and per_channel_wts and param_fp.dim() in [2, 4]

            with torch.no_grad():
                scale, zero_point = _get_quant_shift_params_from_tensor(param_fp, param_meta.num_bits, linear_mode,
                                                                        shift_mode, per_channel=perch)
            setattr(m, param_meta.q_attr_name + '_scale', scale)
            setattr(m, param_meta.q_attr_name + '_zero_point', zero_point)
            out = LinearQuantizeSTE.apply(param_fp, scale, zero_point, True, False)
            return out

        def activation_replace_fn(module, name, qbits_map):
            bits_acts = qbits_map[name].acts
            if bits_acts is None:
                return module
            return FakeShiftQuantizationWrapper(module, bits_acts, linear_mode, shift_mode, ema_decay)

        self.param_quantization_fn = shift_quantize_param

        self.activation_replace_fn = activation_replace_fn
        self.replacement_factory[nn.ReLU] = self.activation_replace_fn

    def _prepare_model_impl(self):
        super(QuantAwareTrainShiftLinearQuantizer, self)._prepare_model_impl()

        if self.quantize_inputs:
            if isinstance(self.model, nn.DataParallel):
                m = self.model.module
            else:
                m = self.model

            m.inputs_quant = FakeShiftQuantization(self.num_bits_inputs, self.linear_mode, self.shift_mode,
                                                   self.decay, dequantize=True, inplace=False)
            m.__class__.original_forward = m.__class__.forward
            m.__class__.forward = inputs_quantize_wrapped_forward

        # Prepare scale and zero point buffers in modules where parameters are being quantized
        # We're calculating "dummy" scale and zero point just to get their dimensions
        for ptq in self.params_to_quantize:
            m = ptq.module
            param_fp = getattr(m, ptq.fp_attr_name)
            perch = not isinstance(m, nn.Embedding) and self.per_channel_wts and param_fp.dim() in [2, 4]
            with torch.no_grad():
                scale, zero_point = _get_quant_shift_params_from_tensor(param_fp, ptq.num_bits, self.linear_mode,
                                                                        self.shift_mode, per_channel=perch)
            m.register_buffer(ptq.q_attr_name + '_scale', torch.ones_like(scale))
            m.register_buffer(ptq.q_attr_name + '_zero_point', torch.zeros_like(zero_point))
