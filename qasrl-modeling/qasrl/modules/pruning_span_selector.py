from typing import Dict, List, TextIO, Optional, Union

from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout, Sequential, ReLU
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
from qasrl.common.span import Span

# from qasrl.metrics.span_metric import SpanMetric

# TODO: fix weighted span selection policy.
# right now it determines the targets. instead it should determine the loss weights.
objective_values = ["binary", "multinomial"]
gold_span_selection_policy_values = ["union", "majority", "weighted"]
# multinomial cannot be used with weighted
class PruningSpanSelector(torch.nn.Module, Registrable):
    def __init__(self,
                 input_dim: int,
                 extra_input_dim: int = 0,
                 span_hidden_dim: int = 100,
                 span_ffnn: FeedForward = None,
                 objective: str = "binary",
                 gold_span_selection_policy: str = "union",
                 pruning_ratio: float = 2.0,
                 skip_metrics_during_training: bool = True,
                 # metric: SpanMetric = SpanMetric(),
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None):
        super(PruningSpanSelector, self).__init__()

        self._input_dim = input_dim
        self._span_hidden_dim = span_hidden_dim
        self._extra_input_dim = extra_input_dim
        self._span_ffnn = span_ffnn
        self._pruning_ratio = pruning_ratio
        self._objective = objective
        self._gold_span_selection_policy = gold_span_selection_policy
        self._skip_metrics_during_training = skip_metrics_during_training

        if objective not in objective_values:
            raise ConfigurationError("QA objective must be one of the following: " + str(qa_objective_values))

        if gold_span_selection_policy not in gold_span_selection_policy_values:
            raise ConfigurationError("QA span selection policy must be one of the following: " + str(qa_objective_values))

        if objective == "multinomial" and gold_span_selection_policy == "weighted":
            raise ConfigurationError("Cannot use weighted span selection policy with multinomial objective.")

        # self._metric = metric

        self._span_hidden = SpanRepAssembly(input_dim, input_dim, self._span_hidden_dim)

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
                num_answers: torch.LongTensor = None, # only needs to be non-None when training with weighted or majority gold span selection policy
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

        (top_span_hidden, top_span_mask,
         top_span_indices, top_span_logits) = self._span_pruner(full_hidden, span_mask.float(), int(self._pruning_ratio * num_tokens))
        top_span_mask = top_span_mask.unsqueeze(-1).float()

        # workaround for https://github.com/allenai/allennlp/issues/1696
        # TODO I think the issue has been fixed and we can remove this?
        if (top_span_logits == float("-inf")).any():
            top_span_logits[top_span_logits == float("-inf")] = -1.

        if answer_spans is not None:
            gold_span_labels = self._get_prediction_map(answer_spans,
                                                       num_tokens, num_answers,
                                                       self._gold_span_selection_policy)
            prediction_mask = batched_index_select(gold_span_labels.unsqueeze(-1),
                                                   top_span_indices)

        top_span_probs = torch.sigmoid(top_span_logits) * top_span_mask
        output_dict = {
            "span_mask": span_mask,
            "top_span_indices": top_span_indices,
            "top_span_mask": top_span_mask,
            "top_span_logits": top_span_logits,
            "top_span_probs": top_span_probs
        }
        if answer_spans is not None:
            loss = F.binary_cross_entropy_with_logits(top_span_logits, prediction_mask,
                                                        weight = top_span_mask, reduction = "sum")
            output_dict["loss"] = loss
        if not (self.training and self._skip_metrics_during_training):
            output_dict = self.decode(output_dict)
            # self._metric(output_dict["spans"], [m["gold_spans"] for m in metadata])
        return output_dict

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "spans" not in output_dict:
            o = output_dict
            spans = self._to_scored_spans(
                o["span_mask"], o["top_span_indices"], o["top_span_mask"], o["top_span_probs"]
            )
            output_dict['spans'] = spans
        return output_dict

    def _to_scored_spans(self, span_mask, top_span_indices, top_span_mask, top_span_probs):
        span_mask = span_mask.data.cpu()
        top_span_indices = top_span_indices.data.cpu()
        top_span_mask = top_span_mask.data.cpu()
        top_span_probs = top_span_probs.data.cpu()
        batch_size, num_spans = span_mask.size()
        top_spans = []
        for b in range(batch_size):
            batch_spans = []
            for start, end, i in self._start_end_range(num_spans):
                batch_spans.append(Span(start, end))
            batch_top_spans = []
            for i in range(top_span_indices.size(1)):
                if top_span_mask[b, i].item() == 1:
                    batch_top_spans.append((batch_spans[top_span_indices[b, i]], top_span_probs[b, i].item()))
            top_spans.append(batch_top_spans)
        return top_spans

    def _start_end_range(self, num_spans):
        n = int(.5 * (math.sqrt(8 * num_spans + 1) -1))

        result = []
        i = 0
        for start in range(n):
            for end in range(start, n):
                result.append((start, end, i))
                i += 1

        return result

    def _get_prediction_map(self, spans, seq_length, num_answerers, span_selection_policy):
        batchsize, num_spans, _ = spans.size()
        span_mask = (spans[:, :, 0] >= 0).view(batchsize, num_spans).long()
        num_labels = int((seq_length * (seq_length+1))/2)
        labels = spans.data.new().resize_(batchsize, num_labels).zero_().float()
        spans = spans.data
        arg_indexes = (2 * spans[:,:,0] * seq_length - spans[:,:,0].float().pow(2).long() + spans[:,:,0]) / 2 + (spans[:,:,1] - spans[:,:,0])
        arg_indexes = arg_indexes * span_mask.data

        for b in range(batchsize):
            for s in range(num_spans):
                if span_mask.data[b, s] > 0:
                    if span_selection_policy == "union":
                        labels[b, arg_indexes[b, s]] = 1
                    else:
                        assert span_selection_policy == "weighted" or span_selection_policy == "majority"
                        labels[b, arg_indexes[b, s]] += 1

        if span_selection_policy == "union":
            return torch.autograd.Variable(labels.float())
        else: # weighted or majority
            if num_answerers is None:
                raise ConfigurationError("Number of answerers must be provided for training the weighted or majority span selection metrics.")
            num_answerers_expanded_to_spans = num_answerers.view(-1, 1).expand(-1, num_labels).float()
            if span_selection_policy == "weighted":
                return torch.autograd.Variable(labels.float() / num_answerers_expanded_to_spans)
            else: # majority
                assert span_selection_policy == "majority"
                return torch.autograd.Variable((labels.float() / num_answerers_expanded_to_spans) >= 0.5).float()

    def get_metrics(self, reset: bool = False):
        return self._metric.get_metric(reset = reset)
