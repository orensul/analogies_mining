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

from qasrl.common.span import Span
from qasrl.metrics.binary_f1 import BinaryF1
from qasrl.metrics.moments_metric import MomentsMetric
from qasrl.modules.span_rep_assembly import SpanRepAssembly
from qasrl.modules.set_classifier.set_classifier import SetClassifier

label_selection_policy_values = ["union", "majority", "weighted"]

@SetClassifier.register("binary")
class SetBinaryClassifier(torch.nn.Module, Registrable):
    def __init__(self,
                 label_selection_policy = "union",
                 skip_metrics_during_training: bool = False,
                 metric: BinaryF1 = BinaryF1()):
        super(SetBinaryClassifier, self).__init__()

        if label_selection_policy not in label_selection_policy_values:
            raise ConfigurationError("Label selection policy must be one of: " + str(label_selection_policy_values))

        self._label_selection_policy = label_selection_policy
        self._skip_metrics_during_training = skip_metrics_during_training
        self._metric = metric
        self._kl_divergence_metric = MomentsMetric()

    def forward(self,  # type: ignore
                logits: torch.LongTensor, # batch_size, set_size, 1
                mask: torch.LongTensor = None, # batch_size, set_size
                label_counts: torch.LongTensor = None, # batch_size, set_size
                num_labelers: torch.LongTensor = None, # batch_size
                metadata = None,
                **kwargs):

        output_dict = {"logits": logits}
        if mask is not None:
            probs = torch.sigmoid(logits) * mask.float()
            output_dict["mask"] = mask
        else:
            probs = torch.sigmoid(logits)
        output_dict["probs"] = probs

        if label_counts is not None:
            float_mask = None if mask is None else mask.float()
            if self._label_selection_policy == "union":
                labels = (label_counts > 0.0).float()
            elif self._label_selection_policy == "majority":
                labels = (label_counts > (num_labelers / 2.0).unsqueeze(-1)).float()
            else:
                assert self._label_selection_policy == "weighted"
                labels = label_counts / num_labelers.unsqueeze(-1)
            cross_entropy = F.binary_cross_entropy_with_logits(
                logits, labels, weight = float_mask, reduction = "none"
            ).sum(1)
            output_dict["loss"] = cross_entropy.sum()
            if not (self.training and self._skip_metrics_during_training):
                self._metric(probs, label_counts > 0.0, mask)
                if self._label_selection_policy == "weighted":
                    inverse_labels = 1.0 - labels
                    gold_entropy = (-labels * labels.log()) - (inverse_labels * inverse_labels.log())
                    gold_entropy[gold_entropy != gold_entropy] = 0.0
                    kl_divergence = cross_entropy - gold_entropy.sum(1)
                    self._kl_divergence_metric(kl_divergence)

        return output_dict

    def get_metrics(self, reset: bool = False):
        metrics_dict = self._metric.get_metric(reset = reset)
        kl_divergence_dict = self._kl_divergence_metric.get_metric(reset = reset)
        if kl_divergence_dict["n"] != 0:
            metrics_dict["KL"] = kl_divergence_dict["mean"]
        return metrics_dict
