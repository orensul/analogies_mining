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
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode
from allennlp.nn.util import batched_index_select
from allennlp.training.metrics import SpanBasedF1Measure

from qasrl.modules.sentence_encoder import SentenceEncoder
from qasrl.modules.slot_sequence_generator import SlotSequenceGenerator
from qasrl.metrics.question_metric import QuestionMetric

@Model.register("qasrl_question")
class QuestionModel(Model):
    def __init__(self, vocab: Vocabulary,
                 sentence_encoder: SentenceEncoder,
                 question_generator: SlotSequenceGenerator,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None):
        super(QuestionModel, self).__init__(vocab, regularizer)
        self._sentence_encoder = sentence_encoder
        self._question_generator = question_generator
        if self._sentence_encoder.get_output_dim() != self._question_generator.get_input_dim():
            raise ConfigurationError(
                ("Input dimension of question generator (%s) must be " % self._question_generator.get_input_dim()) + \
                ("equal to the output dimension of the sentence encoder (%s)." % self._sentence_encoder.get_output_dim()))
        self.metric = QuestionMetric(vocab, self._question_generator.get_slot_names())

    def get_slot_names(self):
        return self._question_generator.get_slot_names()

    @overrides
    def forward(self,
                text: Dict[str, torch.LongTensor],
                predicate_indicator: torch.LongTensor,
                predicate_index: torch.LongTensor,
                **kwargs):
        # slot_name -> Shape: batch_size, 1
        gold_slot_labels = self._get_gold_slot_labels(kwargs)
        if gold_slot_labels is None:
            raise ConfigurationError("QuestionModel requires gold labels for teacher forcing when running forward. "
                                     "You may wish to run beam_decode instead.")
        # Shape: batch_size, num_tokens, self._sentence_encoder.get_output_dim()
        encoded_text, text_mask = self._sentence_encoder(text, predicate_indicator)
        # Shape: batch_size, self._sentence_encoder.get_output_dim()
        pred_rep = batched_index_select(encoded_text, predicate_index).squeeze(1)
        # slot_name -> Shape: batch_size, slot_name_vocab_size
        slot_logits = self._question_generator(pred_rep, **gold_slot_labels)

        batch_size, _ = pred_rep.size()

        # Shape: <scalar>
        slot_nlls, neg_log_likelihood = self._get_cross_entropy(slot_logits, gold_slot_labels)
        self.metric(slot_logits, gold_slot_labels, torch.ones([batch_size]), slot_nlls, neg_log_likelihood)
        return {**slot_logits, "loss": neg_log_likelihood}

    def beam_decode(self,
                    text: Dict[str, torch.LongTensor],
                    predicate_indicator: torch.LongTensor,
                    predicate_index: torch.LongTensor,
                    max_beam_size: int,
                    min_beam_probability: float,
                    clause_mode: bool = False):
        # Shape: batch_size, num_tokens, self._sentence_encoder.get_output_dim()
        encoded_text, text_mask = self._sentence_encoder(text, predicate_indicator)
        # Shape: batch_size, self._sentence_encoder.get_output_dim()
        pred_rep = batched_index_select(encoded_text, predicate_index).squeeze(1)
        return self._question_generator.beam_decode(pred_rep, max_beam_size, min_beam_probability, clause_mode)

    def get_metrics(self, reset: bool = False):
        return self.metric.get_metric(reset=reset)

    def _get_cross_entropy(self, slot_logits, gold_slot_labels):
        slot_xes = {}
        xe = None
        for slot_name in self.get_slot_names():
            slot_xe = F.cross_entropy(slot_logits[slot_name], gold_slot_labels[slot_name].squeeze(-1), reduction = "sum")
            slot_xes[slot_name] = slot_xe
            if xe is None:
                xe = slot_xe
            else:
                xe = xe + slot_xe
        return slot_xes, xe

    def _get_gold_slot_labels(self, instance_slot_labels_dict):
        # each of gold_slot_labels[slot_name] is of
        # Shape: batch_size
        gold_slot_labels = {}
        for slot_name in self.get_slot_names():
            if slot_name in instance_slot_labels_dict and instance_slot_labels_dict[slot_name] is not None:
                gold_slot_labels[slot_name] = instance_slot_labels_dict[slot_name].unsqueeze(-1)
        for slot_name in self.get_slot_names():
            if slot_name not in instance_slot_labels_dict or instance_slot_labels_dict[slot_name] is None:
                gold_slot_labels = None
        return gold_slot_labels
