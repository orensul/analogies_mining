import logging
import re
import math
from typing import Callable, List, Tuple, Type, Iterable, Dict
import itertools
from overrides import overrides

import torch
import torch.nn.init

from allennlp.common import Registrable
from allennlp.common.params import Params
from allennlp.common.checks import ConfigurationError

from allennlp.nn import Initializer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Initializer.register('pretrained_prefixing')
class PretrainedModelInitializer_Prefixing(Initializer):
    def __init__(self,
                 weights_file_path: str,
                 parameter_name_prefix: str = None) -> None:
        self.weights: Dict[str, torch.Tensor] = torch.load(weights_file_path)
        self.parameter_name_prefix = parameter_name_prefix or ""

    @overrides
    def __call__(self, tensor: torch.Tensor, parameter_name: str, **kwargs) -> None:  # type: ignore
        # chop the prefix off of the parameter name
        if parameter_name.startswith(self.parameter_name_prefix):
            parameter_name = parameter_name[len(self.parameter_name_prefix):]

        # If the size of the source and destination tensors are not the
        # same, then we need to raise an error
        source_weights = self.weights[parameter_name]
        if tensor.data.size() != source_weights.size():
            raise ConfigurationError("Incompatible sizes found for parameter %s. "
                                     "Found %s and %s" % (parameter_name,
                                                          tensor.data.size(),
                                                          source_weights.size()))

        # Copy the parameters from the source to the destination
        tensor.data[:] = source_weights[:]
