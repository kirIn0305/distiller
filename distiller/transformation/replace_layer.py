import torch.nn as nn
import numpy as numpy
from enum import Enum
from collections import OrderedDict, namedtuple
import logging

import distiller
import distiller.utils
from .transformer import Transformer
from .transformer import InitilaizationMode
from .transform_modules import PathThrough

class ReplaceLayerTransformer(Transformer):
    def __init__(self, model, optimizer=None,
                 transform_module=None, transform_kernel=None, transform_init_mode=None, overrides=None):
        super(ReplaceLayerTransformer, self).__init__(model, optimizer=optimizer,
                                                     transform_module=transform_module,
                                                     transform_kernel=transform_kernel,
                                                     transform_init_mode=transform_init_mode,
                                                     overrides=overrides)

        def replace_transform_fn(module, name, transform_map, transform_module, transform_init_method=InitilaizationMode.NONE):
            print(name)
            print(transform_map)
            print(transform_module)
            if transform_module is not None:
                if transform_module.get('args') is None:
                    new_module = eval(transform_module['name'])()
                else:
                    new_module = eval(transform_module.name)(*transform_module['args'])

            return new_module

        self.default_replacement_fn = replace_transform_fn