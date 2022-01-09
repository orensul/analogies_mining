from typing import Dict, List, TextIO, Optional, Set, Tuple

from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout, Sequential, ReLU
from torch.autograd import Variable
import torch.nn.functional as F

from allennlp.common import Params, Registrable
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

from qasrl.metrics.binary_f1 import BinaryF1

class SentenceEncoder(torch.nn.Module, Registrable):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 stacked_encoder: Seq2SeqEncoder = None,
                 predicate_feature_dim: int = 0,
                 embedding_dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None):
        super(SentenceEncoder, self).__init__()
        self._text_field_embedder = text_field_embedder
        self._stacked_encoder = stacked_encoder
        self._predicate_feature_dim = predicate_feature_dim
        self._embedding_dropout = Dropout(p = embedding_dropout)

        if self._predicate_feature_dim > 0:
            self._predicate_feature_embedding = Embedding(2, predicate_feature_dim)

        if self._stacked_encoder is not None:
            embedding_dim_with_predicate_feature = self._text_field_embedder.get_output_dim() + self._predicate_feature_dim
            if embedding_dim_with_predicate_feature != self._stacked_encoder.get_input_dim():
                raise ConfigurationError(
                    ("Input dimension of sentence encoder (%s) must be " % self._stacked_encoder.get_input_dim()) + \
                    ("the sum of predicate feature dim and text embedding dim (%s)." % (embedding_dim_with_predicate_feature)))

        self._metric = BinaryF1()

    def get_output_dim(self):
        if self._stacked_encoder is not None:
            return self._stacked_encoder.get_output_dim()
        else:
            return self._text_field_embedder.get_output_dim()

    @overrides
    def forward(self,
                text: Dict[str, torch.LongTensor],
                predicate_indicator: torch.LongTensor):
        # Shape: batch_size, num_tokens, embedding_dim
        embedded_text_input = self._embedding_dropout(self._text_field_embedder(text))
        # Shape: batch_size, num_tokens ?
        text_mask = get_text_field_mask(text)
        if self._predicate_feature_dim > 0:
            # Shape: batch_size, num_tokens, predicate_feature_dim ?
            embedded_predicate_indicator = self._predicate_feature_embedding(predicate_indicator.long())
            # Shape: batch_size, num_tokens, embedding_dim + predicate_feature_dim
            full_embedded_text = torch.cat([embedded_text_input, embedded_predicate_indicator], -1)
        else:
            full_embedded_text = embedded_text_input

        if self._stacked_encoder is not None:
            # Shape: batch_size, num_tokens, encoder_output_dim
            encoded_text = self._stacked_encoder(full_embedded_text, text_mask)
        else:
            encoded_text = full_embedded_text

        return encoded_text, text_mask
