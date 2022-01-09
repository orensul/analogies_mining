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

@Model.register("qasrl_question_to_span")
class QuestionToSpanModel(Model):
    def __init__(self, vocab: Vocabulary,
                 sentence_encoder: SentenceEncoder,
                 question_encoder: SlotSequenceEncoder,
                 span_selector: SpanSelector,
                 classify_invalids: bool = True,
                 invalid_hidden_dim: int = 100,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None):
        super(QuestionToSpanModel, self).__init__(vocab, regularizer)
        self._sentence_encoder = sentence_encoder
        self._question_encoder = question_encoder
        self._span_selector = span_selector
        self._classify_invalids = classify_invalids
        self._invalid_hidden_dim = invalid_hidden_dim

        injected_embedding_dim = self._sentence_encoder.get_output_dim() + self._question_encoder.get_output_dim()
        extra_input_dim = self._span_selector.get_extra_input_dim()
        if injected_embedding_dim != extra_input_dim:
            raise ConfigurationError("Sum of pred rep and question embedding dim %s did not match span selector injection dim of %s" % (injected_embedding_dim, extra_input_dim))

        if self._classify_invalids:
            self._invalid_pred = Sequential(
                Linear(extra_input_dim, self._invalid_hidden_dim),
                ReLU(),
                Linear(self._invalid_hidden_dim, 1))
            self._invalid_metric = BinaryF1()

    def classifies_invalids(self):
        return self._classify_invalids

    def forward(self,  # type: ignore
                text: Dict[str, torch.LongTensor],
                predicate_indicator: torch.LongTensor,
                predicate_index: torch.LongTensor,
                answer_spans: torch.LongTensor = None,
                span_counts: torch.LongTensor = None,
                num_answers: torch.LongTensor = None,
                num_invalids: torch.LongTensor = None,
                metadata = None,
                **kwargs):
        # each of gold_slot_labels[slot_name] is of
        # Shape: batch_size
        slot_labels = self._get_slot_labels(**kwargs)
        if slot_labels is None:
            raise ConfigurationError("QuestionAnswerer must receive question slots as input.")

        encoded_text, text_mask = self._sentence_encoder(text, predicate_indicator)
        pred_rep = batched_index_select(encoded_text, predicate_index).squeeze(1)
        question_encoding = self._question_encoder(pred_rep, slot_labels)
        question_rep = torch.cat([pred_rep, question_encoding], -1)
        output_dict = self._span_selector(
            encoded_text, text_mask,
            extra_input_embedding = question_rep,
            answer_spans = answer_spans,
            span_counts = span_counts,
            num_answers = num_answers,
            metadata = metadata)

        if self._classify_invalids:
            invalid_logits = self._invalid_pred(question_rep).squeeze(-1)
            invalid_probs = torch.sigmoid(invalid_logits)
            output_dict["invalid_prob"] = invalid_probs
            if num_invalids is not None:
                invalid_labels = (num_invalids > 0.0).float()
                invalid_loss = F.binary_cross_entropy_with_logits(invalid_logits, invalid_labels, reduction = "sum")
                output_dict["loss"] += invalid_loss
                self._invalid_metric(invalid_probs, invalid_labels)
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._span_selector.decode(output_dict)

    def get_metrics(self, reset: bool = False):
        span_metrics = self._span_selector.get_metrics(reset = reset)
        if not self._classify_invalids:
            return span_metrics
        else:
            invalid_metrics = self._invalid_metric.get_metric(reset = reset)
            return {
                **{ ("span-%s" % k): v for k, v in span_metrics.items() },
                **{ ("invalid-%s" % k): v for k, v in invalid_metrics.items() }
            }

    def get_slot_names(self):
        return self._question_encoder.get_slot_names()

    def _get_slot_labels(self, **kwargs):
        slot_labels = {}
        for slot_name in self.get_slot_names():
            if slot_name in kwargs and kwargs[slot_name] is not None:
                slot_labels[slot_name] = kwargs[slot_name]
        for slot_name in self.get_slot_names():
            if slot_name not in kwargs or kwargs[slot_name] is None:
                slot_labels = None
        return slot_labels
