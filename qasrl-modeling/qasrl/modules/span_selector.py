
from typing import Dict, List, TextIO, Optional, Union

from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout, ReLU
import torch.nn.functional as F
from torch.nn import Parameter
import math

from allennlp.common import Params, Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TimeDistributed, TextFieldEmbedder, Pruner, FeedForward
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.span_extractors.endpoint_span_extractor import EndpointSpanExtractor
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode
from allennlp.nn.util import batched_index_select
from allennlp.nn.util import masked_log_softmax
from allennlp.training.metrics import SpanBasedF1Measure

from qasrl.modules.span_rep_assembly import SpanRepAssembly
from qasrl.modules.set_classifier.set_classifier import SetClassifier
from qasrl.modules.set_classifier.set_binary_classifier import SetBinaryClassifier
from qasrl.common.span import Span

class SpanSelector(torch.nn.Module, Registrable):
    def __init__(self,
                 input_dim: int,
                 extra_input_dim: int = 0,
                 span_hidden_dim: int = 100,
                 span_ffnn: FeedForward = None,
                 classifier: SetClassifier = SetBinaryClassifier(),
                 span_decoding_threshold: float = 0.05,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None):
        super(SpanSelector, self).__init__()

        self._input_dim = input_dim
        self._extra_input_dim = extra_input_dim
        self._span_hidden_dim = span_hidden_dim
        self._span_ffnn = span_ffnn
        self._classifier = classifier
        self._span_decoding_threshold = span_decoding_threshold

        self._span_hidden = SpanRepAssembly(self._input_dim, self._input_dim, self._span_hidden_dim)
        if self._span_ffnn is not None:
            if self._span_ffnn.get_input_dim() != self._span_hidden_dim:
                raise ConfigurationError(
                    "Span hidden dim %s must match span classifier FFNN input dim %s" % (
                        self._span_hidden_dim, self._span_ffnn.get_input_dim()
                    )
                )
            self._span_scorer = TimeDistributed(
                torch.nn.Sequential(
                    ReLU(),
                    self._span_ffnn,
                    Linear(self._span_ffnn.get_output_dim(), 1)))
        else:
            self._span_scorer = TimeDistributed(
                torch.nn.Sequential(
                    ReLU(),
                    Linear(self._span_hidden_dim, 1)))
        self._span_pruner = Pruner(self._span_scorer)

        if self._extra_input_dim > 0:
            self._extra_input_lin = Linear(self._extra_input_dim, self._span_hidden_dim)

    def get_extra_input_dim(self):
        return self._extra_input_dim

    def forward(self,  # type: ignore
                inputs: torch.LongTensor,
                input_mask: torch.LongTensor,
                extra_input_embedding: torch.LongTensor = None,
                answer_spans: torch.LongTensor = None,
                span_counts: torch.LongTensor = None,
                num_answers: torch.LongTensor = None,
                metadata = None,
                **kwargs):

        if self._extra_input_dim > 0 and extra_input_embedding is None:
            raise ConfigurationError("SpanSelector with extra input configured must receive extra input embeddings.")

        batch_size, num_tokens, _ = inputs.size()
        span_hidden, span_mask = self._span_hidden(inputs, inputs, input_mask, input_mask)

        if self._extra_input_dim > 0:
            full_hidden = self._extra_input_lin(extra_input_embedding).unsqueeze(1) + span_hidden
        else:
            full_hidden = span_hidden

        span_logits = self._span_scorer(full_hidden).squeeze(-1)

        # output_dict = {
        #     "span_mask": span_mask,
        #     "span_logits": span_logits
        # }

        if answer_spans is not None:
            num_gold_spans = answer_spans.size(1)
            span_counts_dist = torch.zeros_like(span_logits)
            for b in range(batch_size):
                for s in range(num_gold_spans):
                    span = answer_spans[b, s]
                    if span[0] > -1:
                        span_index = ((2 * span[0] * num_tokens - span[0].float().pow(2).long() + span[0]) / 2 + (span[1] - span[0])).long()
                        span_counts_dist[b, span_index] = span_counts[b, s]
        else:
            span_counts_dist = None

        return self._classifier(logits = span_logits, mask = span_mask, label_counts = span_counts_dist, num_labelers = num_answers)

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "spans" not in output_dict:
            o = output_dict
            spans = self._to_scored_spans(o["probs"], o["mask"])
            output_dict["spans"] = spans
        return output_dict

    def _to_scored_spans(self, probs, score_mask):
        probs = probs.data.cpu()
        score_mask = score_mask.data.cpu()
        batch_size, num_spans = probs.size()
        spans = []
        for b in range(batch_size):
            batch_spans = []
            for start, end, i in self._start_end_range(num_spans):
                if score_mask[b, i] == 1 and probs[b, i] > self._span_decoding_threshold:
                    batch_spans.append((Span(start, end), probs[b, i].item()))
            spans.append(batch_spans)
        return spans

    def _start_end_range(self, num_spans):
        n = int(.5 * (math.sqrt(8 * num_spans + 1) -1))

        result = []
        i = 0
        for start in range(n):
            for end in range(start, n):
                result.append((start, end, i))
                i += 1

        return result

    def get_metrics(self, reset: bool = False):
        return self._classifier.get_metrics(reset = reset)
