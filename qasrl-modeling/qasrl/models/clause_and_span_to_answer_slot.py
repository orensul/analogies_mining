from typing import Dict, List, TextIO, Optional, Set, Tuple

from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout, Sequential, ReLU
from torch.autograd import Variable
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder, FeedForward
from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.modules.token_embedders import Embedding
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode
from allennlp.nn.util import batched_index_select
from allennlp.training.metrics import SpanBasedF1Measure

from qasrl.modules.sentence_encoder import SentenceEncoder
from qasrl.metrics.binary_f1 import BinaryF1

@Model.register("qasrl_clause_and_span_to_answer_slot")
class ClauseAndSpanToAnswerSlotModel(Model):
    def __init__(self, vocab: Vocabulary,
                 sentence_encoder: SentenceEncoder,
                 qarg_ffnn: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None):
        super(ClauseAndSpanToAnswerSlotModel, self).__init__(vocab, regularizer)
        self._sentence_encoder = sentence_encoder
        self._qarg_ffnn = qarg_ffnn

        self._clause_embedding = Embedding(vocab.get_vocab_size("abst-clause-labels"), self._qarg_ffnn.get_input_dim())
        self._span_extractor = EndpointSpanExtractor(input_dim = self._sentence_encoder.get_output_dim(), combination = "x,y")
        self._span_hidden = TimeDistributed(Linear(2 * self._sentence_encoder.get_output_dim(), self._qarg_ffnn.get_input_dim()))
        self._predicate_hidden = Linear(self._sentence_encoder.get_output_dim(), self._qarg_ffnn.get_input_dim())
        self._qarg_predictor = Linear(self._qarg_ffnn.get_output_dim(), self.vocab.get_vocab_size("qarg-labels"))
        self._metric = BinaryF1()

    @overrides
    def forward(self,
                text: Dict[str, torch.LongTensor],
                predicate_indicator: torch.LongTensor,
                predicate_index: torch.LongTensor,
                qarg_labeled_clauses,
                qarg_labeled_spans,
                qarg_labels = None,
                **kwargs):
        # Shape: batch_size, num_tokens, encoder_output_dim
        encoded_text, text_mask = self._sentence_encoder(text, predicate_indicator)
        # Shape: batch_size, encoder_output_dim
        pred_rep = batched_index_select(encoded_text, predicate_index).squeeze(1)

        batch_size, num_labeled_instances, _ = qarg_labeled_spans.size()
        # Shape: batch_size, num_labeled_instances
        qarg_labeled_mask = (qarg_labeled_spans[:, :, 0] >= 0).squeeze(-1).long()
        if len(qarg_labeled_mask.size()) == 1:
            qarg_labeled_mask = qarg_labeled_mask.unsqueeze(-1)

        # max to prevent the padded labels from messing up the embedding module
        # Shape: batch_size, num_labeled_instances, self._clause_embedding_dim
        input_clauses = self._clause_embedding(qarg_labeled_clauses.max(torch.zeros_like(qarg_labeled_clauses)))

        # Shape: batch_size, num_spans, 2 * encoder_output_projected_dim
        span_embeddings = self._span_extractor(encoded_text, qarg_labeled_spans, text_mask, qarg_labeled_mask)
        # Shape: batch_size, num_spans, self._span_hidden_dim
        input_span_hidden = self._span_hidden(span_embeddings)

        # Shape: batch_size, 1, self._final_input_dim
        pred_embedding = self._predicate_hidden(pred_rep)
        # Shape: batch_size, num_labeled_instances, self._final_input_dim
        qarg_inputs = F.relu(pred_embedding.unsqueeze(1) + input_clauses + input_span_hidden)
        # Shape: batch_size, num_labeled_instances, get_vocab_size("qarg-labels")
        qarg_logits = self._qarg_predictor(self._qarg_ffnn(qarg_inputs))

        final_mask = qarg_labeled_mask.unsqueeze(-1) \
                        .expand(batch_size, num_labeled_instances, self.vocab.get_vocab_size("qarg-labels")) \
                        .float()
        qarg_probs = torch.sigmoid(qarg_logits).squeeze(-1) * final_mask

        output_dict = {"logits": qarg_logits, "probs": qarg_probs}
        if qarg_labels is not None:
            output_dict["loss"] = F.binary_cross_entropy_with_logits(
                qarg_logits, qarg_labels, weight = final_mask, reduction = "sum"
            )
            self._metric(qarg_probs, qarg_labels)
        return output_dict

    def get_metrics(self, reset: bool = False):
        return self._metric.get_metric(reset=reset)
