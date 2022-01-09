from typing import Dict, Union, Sequence, Set, Optional, cast
import logging

from overrides import overrides
import torch

from allennlp.data.fields.field import Field
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class NumberField(Field[torch.Tensor]):

    def __init__(self,
                 value: float) -> None:
        self.value = value

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:  # pylint: disable=no-self-use
        return {}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        # pylint: disable=unused-argument
        return torch.tensor(self.value)

    @overrides
    def empty_field(self):
        return NumberField(0.0)

    def __str__(self) -> str:
        return f"NumberField with value: {self.value}"
