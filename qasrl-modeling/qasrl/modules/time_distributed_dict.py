"""
A wrapper that unrolls the second (time) dimension of a tensor
into the first (batch) dimension, applies some other ``Module``,
and then rolls the time dimension back up.

Modified from AllenNLP to work for modules that output a dict of tensors,
assuming all tensors in the dict have the same shape.
TODO: contribute back to AllenNLP?
"""

import torch


class TimeDistributedDict(torch.nn.Module):
    """
    Given an input shaped like ``(batch_size, time_steps, [rest])`` and a ``Module`` that takes
    inputs like ``(batch_size, [rest])``, ``TimeDistributed`` reshapes the input to be
    ``(batch_size * time_steps, [rest])``, applies the contained ``Module``, then reshapes it back.
    Note that while the above gives shapes with ``batch_size`` first, this ``Module`` also works if
    ``batch_size`` is second - we always just combine the first two dimensions, then split them.
    """
    def __init__(self, module, output_is_dict: bool = True):
        super(TimeDistributedDict, self).__init__()
        self._module = module
        self._output_is_dict = output_is_dict

    def forward(self, **inputs):  # pylint: disable=arguments-differ
        reshaped_inputs = {}
        for k, input_tensor in inputs.items():
            input_size = input_tensor.size()
            if len(input_size) <= 2:
                raise RuntimeError("No dimension to distribute: " + str(input_size))

            # Squash batch_size and time_steps into a single axis; result has shape
            # (batch_size * time_steps, input_size).
            squashed_shape = [-1] + [x for x in input_size[2:]]
            reshaped_inputs[k] = input_tensor.contiguous().view(*squashed_shape)

        if not self._output_is_dict:
            reshaped_outputs = self._module(**reshaped_inputs)

            # Now get the output back into the right shape.
            # (batch_size, time_steps, [hidden_size])
            new_shape = [input_size[0], input_size[1]] + [x for x in reshaped_outputs.size()[1:]]
            outputs = reshaped_outputs.contiguous().view(*new_shape)
        else:
            reshaped_output_dict = self._module(**reshaped_inputs)
            outputs = {}
            for k, v in reshaped_output_dict.items():
                new_shape = [input_size[0], input_size[1]] + [x for x in v.size()[1:]]
                outputs[k] = v.contiguous().view(*new_shape)

        return outputs
