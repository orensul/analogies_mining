from typing import Dict, List, Optional, Set, Tuple

import torch

from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics.metric import Metric

import math

from qasrl.data.util import get_slot_label_namespace

class QuestionMetric(Metric):
    def __init__(self,
            vocabulary: Vocabulary,
            slot_names: List[str]):
        self._vocabulary = vocabulary
        self._slot_names = slot_names

        self.reset()

    def reset(self):
        self._total_questions = 0
        self._questions_correct = 0
        self._slot_correct = { l: 0 for l in self._slot_names }
        self._negative_log_likelihood = 0.
        self._slot_nlls = { l: 0 for l in self._slot_names }

    def __call__(self,
                 slot_logits: Dict[str, torch.Tensor],
                 slot_labels: Dict[str, torch.Tensor],
                 mask: torch.Tensor,
                 slot_nlls: Dict[str, torch.Tensor],
                 negative_log_likelihood: float):
        mask, negative_log_likelihood = self.unwrap_to_tensors(mask.long(), negative_log_likelihood)

        # this metric handles the case where each instance has a sequence of questions
        # as well as the case where each instance has a single question (where the mask is, I assume, all ones).
        has_sequence = len(list(mask.size())) > 1
        if has_sequence:
            batch_size, num_questions = mask.size()
        else:
            batch_size = mask.size(0)

        num_total_questions = mask.sum().item()

        self._total_questions += num_total_questions
        self._negative_log_likelihood += negative_log_likelihood

        # we'll mask out questions as we miss slots to get the full question accuracy
        correct_questions = mask.clone()

        for slot_name in self._slot_names:
            self._slot_nlls[slot_name] += slot_nlls[slot_name]
            # logits Shape: batch_size, slot_name_vocab_size
            # gold_labels Shape: batch_size, 1?
            logits, gold_labels = self.unwrap_to_tensors(slot_logits[slot_name], slot_labels[slot_name])
            # Shape: batch_size, question_length, 1?
            argmax_predictions = logits.argmax(-1)

            if has_sequence:
                for bi in range(batch_size):
                    for qi in range(num_questions):
                        if mask[bi, qi].item() > 0:
                            if argmax_predictions[bi, qi].item() == gold_labels[bi, qi].item():
                                self._slot_correct[slot_name] += 1
                            else:
                                correct_questions[bi, qi] = 0
            else:
                for bi in range(batch_size):
                    if mask[bi].item() > 0:
                        if argmax_predictions[bi].item() == gold_labels[bi].item():
                            self._slot_correct[slot_name] += 1
                        else:
                            correct_questions[bi] = 0


        self._questions_correct += correct_questions.sum().item()

    def get_metric(self, reset=False):

        def get_slot_accuracy(slot_name):
            return self._slot_correct[slot_name] /  self._total_questions
        slot_wise_metrics = {
            **{"%s-acc" % l : get_slot_accuracy(l) for l in self._slot_names},
            **{"%s-pps" % l : math.exp(self._slot_nlls[l] / self._total_questions) for l in self._slot_names}
        }

        avg_slot_accuracy = sum([v for k, v in slot_wise_metrics.items()]) / len(self._slot_names)
        full_question_accuracy = self._questions_correct / self._total_questions
        perplexity_per_question = math.exp(self._negative_log_likelihood / self._total_questions)

        other_metrics = {
            "avg-slot-acc": avg_slot_accuracy,
            "full-question-acc": full_question_accuracy,
            "perplexity-per-question": perplexity_per_question
        }

        if reset:
            self.reset()
        return {**slot_wise_metrics, **other_metrics}

