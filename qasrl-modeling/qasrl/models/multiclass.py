from typing import Dict, List, TextIO, Optional, Set, Tuple

from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout, Sequential, ReLU
from torch.autograd import Variable
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode
from allennlp.nn.util import batched_index_select
from allennlp.training.metrics import SpanBasedF1Measure

from qasrl.modules.sentence_encoder import SentenceEncoder
from qasrl.metrics.binary_f1 import BinaryF1
from qasrl.modules.set_classifier import SetClassifier

@Model.register("qasrl_multiclass")
class MulticlassModel(Model):
    def __init__(self, vocab: Vocabulary,
                 sentence_encoder: SentenceEncoder,
                 label_name: str,
                 label_namespace: str,
                 classifier: SetClassifier,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None):
        super(MulticlassModel, self).__init__(vocab, regularizer)
        self._sentence_encoder = sentence_encoder
        self._label_name = label_name
        self._label_namespace = label_namespace
        self._classifier = classifier
        self._final_pred = Linear(self._sentence_encoder.get_output_dim(), self.vocab.get_vocab_size(self._label_namespace))
        self._metric = BinaryF1()

    @overrides
    def forward(self,
                text: Dict[str, torch.LongTensor],
                predicate_indicator: torch.LongTensor,
                predicate_index: torch.LongTensor,
                **kwargs):
        # Shape: batch_size, num_tokens, self._sentence_encoder.get_output_dim()
        encoded_text, text_mask = self._sentence_encoder(text, predicate_indicator)
        # Shape: batch_size, encoder_output_dim
        pred_rep = batched_index_select(encoded_text, predicate_index).squeeze(1)
        # Shape: batch_size, get_vocab_size(self._label_namespace)
        logits = self._final_pred(pred_rep)
        return self._classifier(logits, None, kwargs.get(self._label_name), kwargs.get(self._label_name + "_counts"))

    def get_metrics(self, reset: bool = False):
        return self._classifier.get_metric(reset = reset)
