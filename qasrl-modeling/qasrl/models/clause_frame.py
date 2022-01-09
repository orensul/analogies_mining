from typing import Dict, List, TextIO, Optional, Set, Tuple

from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout, Sequential, ReLU
from torch.nn import Parameter
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
from qasrl.metrics.binary_f1 import BinaryF1
from qasrl.metrics.moments_metric import MomentsMetric
from qasrl.util.sparsemax import sparsemax

@Model.register("qasrl_clause_frame")
class ClauseFrameModel(Model):
    def __init__(self, vocab: Vocabulary,
                 sentence_encoder: SentenceEncoder,
                 num_frames: int = 100,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None):
        super(ClauseFrameModel, self).__init__(vocab, regularizer)
        self._num_frames = num_frames
        self._num_clauses = vocab.get_vocab_size("abst-clause-labels")
        self._sentence_encoder = sentence_encoder
        self._frames_matrix = Parameter(data = torch.zeros([self._num_frames, self._num_clauses], dtype = torch.float32))
        self._frame_pred = Linear(self._sentence_encoder.get_output_dim(), self._num_frames)
        self._metric = BinaryF1()
        self._kl_divergence_metric = MomentsMetric()

        initializer(self)

    # TODO figure out how to use this with a null input / maybe to have logits to reuse the setclassifier one
    @overrides
    def forward(self,
                text: Dict[str, torch.LongTensor],
                predicate_indicator: torch.LongTensor,
                predicate_index: torch.LongTensor,
                clause_dist: torch.FloatTensor = None,
                **kwargs):
        # Shape: batch_size, num_tokens, self._sentence_encoder.get_output_dim()
        encoded_text, text_mask = self._sentence_encoder(text, predicate_indicator)
        # Shape: batch_size, encoder_output_dim
        pred_rep = batched_index_select(encoded_text, predicate_index).squeeze(1)
        # Shape: batch_size, get_vocab_size(self._label_namespace)
        frame_logits = self._frame_pred(pred_rep)
        frame_probs = F.softmax(frame_logits, dim = 1)
        frames = F.softmax(self._frames_matrix, dim = 1)
        clause_probs = torch.matmul(frame_probs, frames)
        clause_log_probs = clause_probs.log()
        output_dict = { "probs": frame_probs }
        # TODO figure out how to do this with logits
        # TODO figure out how to handle the null case
        if clause_dist is not None and clause_dist.sum().item() > 0.1:
            gold_clause_probs = F.normalize(clause_dist.float(), p = 1, dim = 1)
            cross_entropy = torch.sum(-gold_clause_probs * clause_log_probs, 1)
            output_dict["loss"] = torch.mean(cross_entropy)
            gold_entropy = -gold_clause_probs * gold_clause_probs.log() # entropy summands
            gold_entropy[gold_entropy != gold_entropy] = 0.0 # zero out nans
            gold_entropy = torch.sum(gold_entropy, dim = 1) # compute sum for per-batch-item entropy
            kl_divergence = cross_entropy - gold_entropy
            self._metric(clause_probs, clause_dist > 0.0)
            self._kl_divergence_metric(kl_divergence)
        return output_dict

    def get_metrics(self, reset: bool = False):
        metrics_dict = self._metric.get_metric(reset = reset)
        kl_divergence_dict = self._kl_divergence_metric.get_metric(reset = reset)
        if kl_divergence_dict["n"] != 0:
            metrics_dict["KL"] = kl_divergence_dict["mean"]
        return metrics_dict
