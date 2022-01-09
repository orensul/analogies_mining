from typing import Dict, List, TextIO, Optional, Set, Tuple

from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout
from torch.autograd import Variable
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.span_extractors.endpoint_span_extractor import EndpointSpanExtractor
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode
from allennlp.nn.util import batched_index_select
from allennlp.training.metrics import SpanBasedF1Measure

from qasrl.metrics.question_metric import QuestionMetric
from qasrl.modules.slot_sequence_generator import SlotSequenceGenerator
from qasrl.modules.sentence_encoder import SentenceEncoder
from qasrl.modules.time_distributed_dict import TimeDistributedDict

@Model.register("qasrl_span_to_question")
class SpanToQuestionModel(Model):
    def __init__(self, vocab: Vocabulary,
                 sentence_encoder: SentenceEncoder,
                 question_generator: SlotSequenceGenerator,
                 inject_predicate: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None):
        super(SpanToQuestionModel, self).__init__(vocab, regularizer)
        self._sentence_encoder = sentence_encoder
        self._question_generator = question_generator
        self._inject_predicate = inject_predicate

        self._time_distributed_question_generator = TimeDistributedDict(self._question_generator, output_is_dict = True)
        self._span_extractor = EndpointSpanExtractor(self._sentence_encoder.get_output_dim(), combination="x,y")

        question_input_dim = (3 * self._sentence_encoder.get_output_dim()) if self._inject_predicate else (2 * self._sentence_encoder.get_output_dim())
        if question_input_dim != self._question_generator.get_input_dim():
            raise ConfigurationError(
                ("Input dimension of question generator (%s) must " % self._question_generator.get_input_dim()) + \
                ("equal the span embedding dimension (plus predicate representation if necessary) (%s)." % question_input_dim))
        self._metric = QuestionMetric(vocab, self._question_generator.get_slot_names())

    @overrides
    def forward(self,
                text: Dict[str, torch.LongTensor],
                predicate_indicator: torch.LongTensor,
                predicate_index: torch.LongTensor,
                answer_spans: torch.LongTensor,
                **kwargs):
        question_inputs, span_mask = self._get_question_inputs(text, predicate_indicator, predicate_index, answer_spans)
        gold_slot_labels = self._get_gold_slot_labels(span_mask, **kwargs)
        if gold_slot_labels is not None:
            slot_logits = self._time_distributed_question_generator(**{"inputs": question_inputs, **gold_slot_labels})
            slot_nlls, neg_log_likelihood = self._get_total_cross_entropy(slot_logits, gold_slot_labels, span_mask)
            self._metric(slot_logits, gold_slot_labels, span_mask, slot_nlls, neg_log_likelihood)
            return {**slot_logits, "span_mask": span_mask, "loss": neg_log_likelihood}
        else:
            raise ConfigurationError("AfirstQuestionGenerator requires gold labels for teacher forcing when running forward. "
                                     "You may wish to run beam_decode instead.")

    def beam_decode(self,
                    text: Dict[str, torch.LongTensor],
                    predicate_indicator: torch.LongTensor,
                    predicate_index: torch.LongTensor,
                    answer_spans: torch.LongTensor,
                    max_beam_size: int,
                    min_beam_probability: float):
        # Shape: 1, num_spans, question generator input dim
        question_inputs, _ = self._get_question_inputs(text, predicate_indicator, predicate_index, answer_spans)
        batch_size, num_spans, _ = question_inputs.size()
        if batch_size > 1:
            raise ConfigurationError("Must have a batch size of 1 for beam decoding (had batch size %s)" % (num_spans, batch_size))
        question_inputs = question_inputs.squeeze(0)
        return [self._question_generator.beam_decode(
            question_inputs[i].unsqueeze(0), max_beam_size, min_beam_probability
        ) for i in range(num_spans)]

    def get_slot_names(self):
        return self._question_generator.get_slot_names()

    def get_metrics(self, reset: bool = False):
        return self._metric.get_metric(reset=reset)

    def _get_total_cross_entropy(self, slot_logits, gold_slot_labels, span_mask):
        loss = 0.
        slot_xes = {}
        for n in self._question_generator.get_slot_names():
            slot_loss = sequence_cross_entropy_with_logits(
                slot_logits[n], gold_slot_labels[n], span_mask.unsqueeze(-1).float(),
                average = None
            ).sum()
            slot_xes[n] = slot_loss
            loss += slot_loss
        return slot_xes, loss

    def _get_gold_slot_labels(self, mask, **kwargs):
        slot_labels = {}
        for slot_name in self._question_generator.get_slot_names():
            if slot_name in kwargs and kwargs[slot_name] is not None:
                slot_labels[slot_name] = (kwargs[slot_name] * mask).unsqueeze(-1)
        if len(slot_labels) == 0:
            slot_labels = None
        return slot_labels

    def _get_question_inputs(self, text, predicate_indicator, predicate_index, answer_spans):
        encoded_text, text_mask = self._sentence_encoder(text, predicate_indicator)
        span_mask = (answer_spans[:, :, 0] >= 0).long()
        # Shape: batch_size, num_spans, 2 * self._sentence_encoder.get_output_dim()
        span_reps = self._span_extractor(encoded_text, answer_spans, sequence_mask = text_mask, span_indices_mask = span_mask)
        batch_size, _, encoding_dim = encoded_text.size()
        num_spans = span_reps.size(1)
        if self._inject_predicate:
            pred_rep_expanded = batched_index_select(encoded_text, predicate_index) \
                                .expand(batch_size, num_spans, encoding_dim)
            question_inputs = torch.cat([pred_rep_expanded, span_reps], -1)
        else:
            question_inputs = span_reps
        return question_inputs, span_mask
