from typing import Dict, List, TextIO, Optional, Union

import torch

from allennlp.common import Registrable

class SetClassifier(torch.nn.Module, Registrable):
    def __init__(self):
        super(SetClassifier, self).__init__()

    def forward(self,  # type: ignore
                logits: torch.LongTensor, # batch_size, set_size, 1
                mask: torch.LongTensor = None, # batch_size, set_size
                label_counts: torch.LongTensor = None, # batch_size, set_size
                num_labelers: torch.LongTensor = None): # batch_size
        raise NotImplementedError

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return output_dict

    def get_metrics(self, reset: bool = False):
        return self._metric.get_metric(reset = reset)
