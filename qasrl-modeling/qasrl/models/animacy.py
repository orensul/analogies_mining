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

@Model.register("qasrl_animacy")
class AnimacyModel(Model):
    def __init__(self, vocab: Vocabulary,
                 sentence_encoder: SentenceEncoder,
                 animacy_ffnn: FeedForward,
                 inject_predicate: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None):
        super(AnimacyModel, self).__init__(vocab, regularizer)
        self._sentence_encoder = sentence_encoder
        self._animacy_ffnn = animacy_ffnn
        self._inject_predicate = inject_predicate

        self._span_extractor = EndpointSpanExtractor(input_dim = self._sentence_encoder.get_output_dim(), combination = "x,y")
        prediction_input_dim = (3 * self._sentence_encoder.get_output_dim()) if self._inject_predicate else (2 * self._sentence_encoder.get_output_dim())
        self._animacy_pred = TimeDistributed(
            Sequential(
                Linear(prediction_input_dim, self._animacy_ffnn.get_input_dim()),
                ReLU(),
                self._animacy_ffnn,
                Linear(self._animacy_ffnn.get_output_dim(), 1)))
        self._metric = BinaryF1()

    @overrides
    def forward(self,
                text: Dict[str, torch.LongTensor],
                predicate_indicator: torch.LongTensor,
                predicate_index: torch.LongTensor,
                animacy_spans,
                animacy_labels = None,
                **kwargs):
        # Shape: batch_size, num_tokens, encoder_output_dim
        encoded_text, text_mask = self._sentence_encoder(text, predicate_indicator)

        # Shape: batch_size, num_labeled_instances
        animacy_mask = (animacy_spans[:, :, 0] >= 0).squeeze(-1).float()
        if len(animacy_mask.size()) == 1:
            animacy_mask = animacy_mask.unsqueeze(-1)

        # Shape: batch_size, num_spans, 2 * encoder_output_projected_dim
        span_embeddings = self._span_extractor(encoded_text, animacy_spans, text_mask, animacy_mask.long())
        batch_size, num_spans, _ = span_embeddings.size()

        if self._inject_predicate:
            expanded_pred_embedding = batched_index_select(encoded_text, predicate_index) \
                                      .expand(batch_size, num_spans, self._sentence_encoder.get_output_dim())
            input_embeddings = torch.cat([expanded_pred_embedding, span_embeddings], -1)
        else:
            input_embeddings = span_embeddings

        # Shape: batch_size, num_labeled_instances, 1
        animacy_logits = self._animacy_pred(input_embeddings).squeeze(-1)
        animacy_probs = torch.sigmoid(animacy_logits) * animacy_mask

        output_dict = {"logits": animacy_logits, "probs": animacy_probs}
        if animacy_labels is not None:
            output_dict["loss"] = F.binary_cross_entropy_with_logits(
                animacy_logits, animacy_labels.float(), weight = animacy_mask, reduction = "sum"
            )
            self._metric(animacy_probs, animacy_labels, animacy_mask.long())
        return output_dict

    def get_metrics(self, reset: bool = False):
        return self._metric.get_metric(reset=reset)
