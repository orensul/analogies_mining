from typing import Dict, Optional

import torch
from torch.nn.modules import Dropout

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import batched_index_select

from qasrl.modules.sentence_encoder import SentenceEncoder
from qasrl.modules.span_selector import SpanSelector

# Slightly modified re-implementation of Nicholas's span detector
@Model.register("qasrl_span")
class SpanModel(Model):
    def __init__(self, vocab: Vocabulary,
                 sentence_encoder: SentenceEncoder,
                 span_selector: SpanSelector,
                 inject_predicate: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None):
        super(SpanModel, self).__init__(vocab, regularizer)
        self._sentence_encoder = sentence_encoder
        self._span_selector = span_selector
        self._inject_predicate = inject_predicate

        if self._inject_predicate and self._span_selector.get_extra_input_dim() != self._sentence_encoder.get_output_dim():
            raise ConfigurationError(
                "When using inject_predicate, span selector's injection dim %s must match sentence encoder output dim %s" % (
                    self._span_selector.get_extra_input_dim(), self._sentence_encoder.get_output_dim()
                )
            )
        if (not self._inject_predicate) and self._span_selector.get_extra_input_dim() > 0:
            raise ConfigurationError(
                "When not using inject_predicate, span selector's injection dim %s must be 0" % (
                    self._span_selector.get_extra_input_dim()
                )
            )

    def forward(self,  # type: ignore
                text: Dict[str, torch.LongTensor],
                predicate_indicator: torch.LongTensor = None,
                predicate_index: torch.LongTensor = None,
                answer_spans: torch.LongTensor = None,
                span_counts: torch.LongTensor = None,
                num_answers: torch.LongTensor = None,
                metadata = None,
                **kwargs):
        encoded_text, text_mask = self._sentence_encoder(text, predicate_indicator)
        extra_input = batched_index_select(encoded_text, predicate_index).squeeze(1) if self._inject_predicate else None
        return self._span_selector(
            encoded_text, text_mask,
            extra_input_embedding = extra_input,
            answer_spans = answer_spans,
            span_counts = span_counts,
            num_answers = num_answers,
            metadata = metadata)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._span_selector.decode(output_dict)

    def get_metrics(self, reset: bool = False):
        return self._span_selector.get_metrics(reset = reset)
