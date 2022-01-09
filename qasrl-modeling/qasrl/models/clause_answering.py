from typing import Dict, List, TextIO, Optional, Union

from overrides import overrides

import torch
from torch.nn.modules import Dropout, Sequential, Linear, ReLU
import torch.nn.functional as F

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import batched_index_select

from qasrl.metrics.binary_f1 import BinaryF1

from qasrl.modules.slot_sequence_encoder import SlotSequenceEncoder
from qasrl.modules.span_selector import SpanSelector
from qasrl.modules.sentence_encoder import SentenceEncoder

@Model.register("qasrl_clause_answering")
class ClauseAnsweringModel(Model):
    def __init__(self, vocab: Vocabulary,
                 sentence_encoder: SentenceEncoder,
                 clause_embedding_dim: int,
                 slot_embedding_dim: int,
                 span_selector: SpanSelector,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None):
        super(ClauseAnsweringModel, self).__init__(vocab, regularizer)
        self._sentence_encoder = sentence_encoder
        self._clause_embedding_dim = clause_embedding_dim
        self._slot_embedding_dim = slot_embedding_dim
        self._span_selector = span_selector
        self._question_embedding_dim = span_selector.get_extra_input_dim()

        self._clause_embedding = Embedding(vocab.get_vocab_size("clause-template-labels"), clause_embedding_dim)
        self._slot_embedding = Embedding(vocab.get_vocab_size("answer-slot-labels"), slot_embedding_dim)

        self._combined_embedding_dim = self._sentence_encoder.get_output_dim() + \
                                       self._clause_embedding_dim + \
                                       self._slot_embedding_dim
        self._question_projection = Linear(self._combined_embedding_dim, self._question_embedding_dim)

        if self._question_embedding_dim == 0:
            raise ConfigurationError("Question embedding dim (span selector extra input dim) cannot be 0")

    def forward(self,  # type: ignore
                text: Dict[str, torch.LongTensor],
                predicate_indicator: torch.LongTensor,
                predicate_index: torch.LongTensor,
                clause: torch.LongTensor,
                answer_slot: torch.LongTensor,
                answer_spans: torch.LongTensor = None,
                span_counts: torch.LongTensor = None,
                num_answers: torch.LongTensor = None,
                metadata = None,
                **kwargs):
        embedded_clause = self._clause_embedding(clause)
        embedded_slot = self._slot_embedding(answer_slot)
        encoded_text, text_mask = self._sentence_encoder(text, predicate_indicator)
        pred_rep = batched_index_select(encoded_text, predicate_index).squeeze(1)
        combined_embedding = torch.cat([embedded_clause, embedded_slot, pred_rep], -1)
        question_embedding = self._question_projection(combined_embedding)
        return self._span_selector(
            encoded_text, text_mask,
            extra_input_embedding = question_embedding,
            answer_spans = answer_spans,
            span_counts = span_counts,
            num_answers = num_answers,
            metadata = metadata)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._span_selector.decode(output_dict)

    def get_metrics(self, reset: bool = False):
        return self._span_selector.get_metrics(reset = reset)
