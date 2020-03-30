from collections import namedtuple, OrderedDict
from enum import Enum
import re
import torch
import torch.nn as nn
import logging
import distiller
import copy

msglogger = logging.getLogger()
TransLayer = namedtuple('TransLayer', ['module', 'kernel', 'init_mode'])


class InitilaizationMode(Enum):
    # Default Initialization Method of Pytorch
    NONE = 0


class Transformer(object):
    """ Base Class for all transformers.

    Args:
        model (torch.nn.Module): The model to be transformed
        optimizer (torch.optim.Optimizer): An optimizer instance, required in cases where the transformer is going
        overrides (OrderedDict): Dictionary mapping regular expressions of layer name patterns to dictionary with overrides of default values.
    Note:
        The `overrides` dictionary assumes the keys are *not* the module names in the
        `nn.DataParallel` case - i.e. without the `module.` prefix. e.g.:
        module.conv1 -> OrderedDict([('conv1', OrderedDict(...))])
    """
    def __init__(self, model, optimizer=None,
                 transform_module=None, transform_kernel=None, transform_init_mode=None,
                 overrides=None):
        if overrides is None:
            #overrides = OrderedDict()
            #TODO add default layer of transform mode
            raise TypeError('overrides must be an instance of collections.OrderedDict')
        if not isinstance(overrides, OrderedDict):
            raise TypeError('overrides must be an instance of collections.OrderedDict or None')

        self.default_transform = TransLayer(module=transform_module, kernel=transform_kernel, init_mode=transform_init_mode)
        self.overrides = overrides

        self.model = model
        self.optimizer = optimizer
        self.model.transformer_metadata = {'type': type(self),
                                           'params': {'transform_module': transform_module,
                                                      'transform_kernel': transform_kernel,
                                                      'transform_init_mode': transform_init_mode,
                                                      'overrides': copy.deepcopy(overrides)}}

        for k, v in self.overrides.items():
            transform = TransLayer(module=v.pop('module', self.default_transform.module),
                                   kernel=v.pop('kernel', self.default_transform.kernel),
                                   init_mode=v.pop('init_mode', self.default_transform.init_mode))
            v['transform'] = transform

        # Prepare explicit mapping from each layer to TransLayer based on default + overrides
        patterns = []
        regex_overrides = None
        if overrides:
            patterns = list(overrides.keys())
            regex_overrides_str = '|'.join(['(^{0}$)'.format(pattern) for pattern in patterns])
            regex_overrides = re.compile(regex_overrides_str)

        self.module_transform_map = {}  # type: OrderedDict[str, TransLayer]
        self.module_overrides_map = {}

        for module_full_name, module in model.named_modules():
            # Need to account for scenario where model is parallelized with DataParallel, which wraps the original
            # module with a wrapper module called 'module' :)
            name_to_match = module_full_name.replace('module.', '', 1)
            transform = self.default_transform
            override_entry = self.overrides.get(name_to_match, OrderedDict())
            if regex_overrides:
                m_overrides = regex_overrides.match(name_to_match)
                if m_overrides:
                    group_idx = 0
                    groups = m_overrides.groups()
                    while groups[group_idx] is None:
                        group_idx += 1
                    override_entry = copy.deepcopy(override_entry or self.overrides[patterns[group_idx]])
                    transform = override_entry.pop('transform', self.default_transform)

            self._add_transform_entry(module_full_name, type(module), transform)
            self._add_override_entry(module_full_name, override_entry)

        # Mapping from module type to function generating a replacement module suited for transform
        # To be populated by child classes
        # Unspecified layer types return None by default.
        self.replacement_factory = OrderedDict([(nn.Identity, None)])
        self.default_replacement_fn = None
        self.replacement_blacklist = []

        # A dictionary of replaced modules and their respective names.
        self.modules_processed = OrderedDict()
        self.modules_processed_args = OrderedDict()

        self.prepared = False

    def _add_transform_entry(self, module_name, module_type, transform):
        self.module_transform_map[module_name] = transform

    def _add_override_entry(self, module_name, entry):
        self.module_overrides_map[module_name] = entry

    def prepare_model(self, dummy_input=None):
        """
        Traverses the model and replaces sub-modules with another layer module
        """
        if self.prepared:
            raise RuntimeError('prepare_model can be called only once')

        msglogger.info('Preparing model for transformation using {0}'.format(self.__class__.__name__))

        model_device = distiller.model_device(self.model)

        self._pre_prepare_model(dummy_input)

        self._pre_process_container(self.model)

        # If an optimizer was passed, assume we need to update it
        # if self.optimizer:
        #     for pg in self._get_new_optimizer_params_groups():
        #         self.optimizer.add_param_group(pg)

        # Re-transfer model to the device it was on, in case the transformer created new parameters/buffers
        self.model.to(model_device)
        self.prepared = True
        msglogger.info('Transformed model:\n\n{0}\n'.format(self.model))

    def _pre_prepare_model(self, dummy_input):
        pass

    def _pre_process_container(self, container, prefix=''):
        def replace_msg(module_name, modules=None):
            msglogger.debug('Module ' + module_name)
            if modules:
                msglogger.debug('\tReplacing: {}.{}'.format(modules[0].__module__, modules[0].__class__.__name__))
                msglogger.debug('\tWith:      {}.{}'.format(modules[1].__module__, modules[1].__class__.__name__))
            else:
                msglogger.debug('\tSkipping')

        # Iterate through model, insert transrom functions as appropriate
        for name, module in container.named_children():
            full_name = prefix + name
            if isinstance(module, tuple(self.replacement_blacklist)):
                replace_msg(full_name)
                continue
            #TODO consider to add procecced
            #current_transform = self.module_transform_map[full_name]
            #TODO apply all layers of specified nn.Module(eg. select nn.Conv2d and apply only nn.Conv2d)
            #if current_transform.module is None and current_transform.kernel is None and current_transform.init_mode is None:
            #    replace_msg(full_name)
            #    self.modules_processed[module] = full_name, None
            #else:
            # We use a type hint comment to let IDEs know replace_fn is a function
            replace_fn = self.replacement_factory.get(type(module), self.default_replacement_fn)
            # If the replacement function wasn't specified - continue without replacing this module.
            if replace_fn is not None:
                valid_kwargs, invalid_kwargs = distiller.filter_kwargs(self.module_overrides_map[full_name], replace_fn)
                if invalid_kwargs:
                    raise TypeError("""Transformer of type %s doesn't accept \"%s\" as override arguments for %s. Allowed kwargs: %s"""
                                    % (type(self), list(invalid_kwargs), type(module), list(valid_kwargs)))
                if valid_kwargs:
                    new_module = replace_fn(module, full_name, self.module_transform_map[full_name], **valid_kwargs)
                    if new_module != module:
                        replace_msg(full_name, (module, new_module))
                        # Add to history of prepared submodules
                        self.modules_processed[module] = full_name, new_module
                        # To allow recreating this wrapper later on
                        valid_args = full_name, copy.deepcopy(self.module_transform_map[full_name])
                        self.modules_processed_args[full_name] = valid_args, valid_kwargs
                        setattr(container, name, new_module)

                        ## If a "leaf" module was replaced by a container, add the new layers to the QBits mapping
                        #if not distiller.has_children(module) and distiller.has_children(new_module):
                        #    for sub_module_name, sub_module in new_module.named_modules():
                        #        self._add_transform_entry(full_name + '.' + sub_module_name, type(sub_module), current_transform)
                        #    self.module_transorm_map[full_name] = TransLayer(acts=current_qbits.acts, wts=None, bias=None)
                else:
                    replace_msg(full_name)
                    self.modules_processed[module] = full_name, None

            if distiller.has_children(module):
                # For container we call recursively
                self._pre_process_container(module, full_name + '.')