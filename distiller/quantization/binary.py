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

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizer import Quantizer
from .q_utils import *
import logging
msglogger = logging.getLogger()

###
# Binary-based quantization (e.g. BinaryConnect, Binaraized Networks)
###

def inputs_quantize_wrapped_forward(self, input):
    input = self.inputs_quant(input)
    return self.original_forward(input)


def binary_quantize(input, inplace=False):
    if inplace:
        input.sign_()
        return input
    return torch.sign(input)


class BinaryQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, inplace=False):
        if inplace:
            ctx.mark_dirty(input)
        output = binary_quantize(input, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output, None, None, None, None


class BinaryQuantization(nn.Module):
    def __init__(self, inplace=False):
        super(BinaryQuantization, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        input = BinaryQuantizeSTE.apply(input, self.inplace)
        return input

    def extra_repr(self):
        return f'Binary Quantization, inplace={self.inplace}'


class BinaryQuantizationWrapper(nn.Module):
    def __init__(self, wrapped_module, inplace=False):
        super(BinaryQuantizationWrapper, self).__init__()
        self.wrapped_module = wrapped_module
        self.binary_q = BinaryQuantization(inplace)

    def forward(self, *input):
        res = self.wrapped_module(*input)
        res = self.binary_q(res)
        return res


class BinaryQuantizer(Quantizer):
    def __init__(self, model, optimizer=None, bits_activations=32, bits_weights=32, bits_bias=32, overrides=None, per_channel_wts=False, quantize_inputs=False, num_bits_inputs=None):
        super(BinaryQuantizer, self).__init__(model, optimizer=optimizer,
                                              bits_activations=bits_activations,
                                              bits_weights=bits_weights,
                                              bits_bias=bits_bias,
                                              overrides=overrides,
                                              train_with_fp_copy=True)

        self.quantize_inputs = quantize_inputs
        if num_bits_inputs is not None:
            self.num_bits_inputs = num_bits_inputs
        else:
            self.num_bits_inputs = bits_activations
        self.per_channel_wts = per_channel_wts

        def binary_quantize_param(param_fp, param_meta):
            m = param_meta.module

            #perch = not isinstance(m, nn.Embedding) and per_channel_wts and param_fp.dim() in [2, 4]
            #with torch.no_grad():
            #    scale, zero_point = _get_quant_params_from_tensor(param_fp, param_meta.num_bits, mode,
            out = BinaryQuantizeSTE.apply(param_fp, False)
            return out

        def activation_replace_fn(module, name, qbits_map):
            bits_acts = qbits_map[name].bits_acts
            if bits_acts is None:
                return module
            return BinaryQuantizationWrapper(module)

        self.param_quantization_fn = binary_quantize_param
        self.activation_replace_fn = activation_replace_fn
        self.replacement_factory[nn.ReLU] = self.activation_replace_fn

    def _post_prepare_model(self):
        if self.quantize_inputs:
            if isinstance(self.model, nn.DataParallel):
                m = self.model.module
            else:
                m = self.model

            m.inputs_quant = FakeLinearQuantization(self.num_bits_inputs, self.mode, self.decay,
                                                    dequantize=True, inplace=False)
            m.__class__.original_forward = m.__class__.forward
            m.__class__.forward = inputs_quantize_wrapped_forward
