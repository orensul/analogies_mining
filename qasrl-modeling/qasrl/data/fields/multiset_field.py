from typing import Dict, Union, Sequence, Set, Optional, cast
import logging
from collections import Counter

from overrides import overrides
import torch

from allennlp.data.fields.field import Field
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class MultisetField(Field[torch.Tensor]):
    _already_warned_namespaces: Set[str] = set()

    def __init__(self,
                 labels, # Counter of Union[str, int]
                 label_namespace: str = 'labels',
                 skip_indexing: bool = False,
                 num_labels: Optional[int] = None) -> None:
        self.labels = labels
        self._label_namespace = label_namespace
        self._label_ids = None
        self._maybe_warn_for_namespace(label_namespace)
        self._num_labels = num_labels

        if skip_indexing:
            if not all(isinstance(label, int) for label in labels):
                raise ConfigurationError("In order to skip indexing, your labels must be integers. "
                                         "Found labels = {}".format(labels))
            if not num_labels:
                raise ConfigurationError("In order to skip indexing, num_labels can't be None.")

            if not all(cast(int, label) < num_labels for label in labels):
                raise ConfigurationError("All labels should be < num_labels. "
                                         "Found num_labels = {} and labels = {} ".format(num_labels, labels))

            self._label_ids = labels
        else:
            if not all(isinstance(label, str) for label in labels):
                raise ConfigurationError("MultiSetField expects string labels if skip_indexing=False. "
                                         "Found labels: {}".format(labels))

    def _maybe_warn_for_namespace(self, label_namespace: str) -> None:
        if not (label_namespace.endswith("labels") or label_namespace.endswith("tags")):
            if label_namespace not in self._already_warned_namespaces:
                logger.warning("Your label namespace was '%s'. We recommend you use a namespace "
                               "ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by "
                               "default to your vocabulary.  See documentation for "
                               "`non_padded_namespaces` parameter in Vocabulary.",
                               self._label_namespace)
                self._already_warned_namespaces.add(label_namespace)

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        if self._label_ids is None:
            for label in self.labels:
                counter[self._label_namespace][label] += self.labels[label]  # type: ignore

    @overrides
    def index(self, vocab: Vocabulary):
        if self._label_ids is None:
            self._label_ids = Counter()
            for label in self.labels:
                label_id = vocab.get_token_index(label, self._label_namespace)
                self._label_ids[label_id] += self.labels[label]
        if not self._num_labels:
            self._num_labels = vocab.get_vocab_size(self._label_namespace)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:  # pylint: disable=no-self-use
        return {}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        # pylint: disable=unused-argument

        return torch.tensor([self._label_ids[i] for i in range(self._num_labels)])
        # tensor = torch.zeros(self._num_labels)  # vector of zeros
        # if self._label_ids:
        #     tensor.scatter_(0, torch.LongTensor(self._label_ids), 1)

        # return tensor

    @overrides
    def empty_field(self):
        return MultiSetField(Counter(), self._label_namespace, skip_indexing = self._num_labels is not None, num_labels = self._num_labels)

    def __str__(self) -> str:
        return f"MultiSetField with labels: {self.labels} in namespace: '{self._label_namespace}'.'"

