 # completely ridiculous hack to import stuff properly. somebody save me from myself
import importlib
from allennlp.common.util import import_submodules
importlib.invalidate_caches()
import sys
sys.path.append(".")
import_submodules("qasrl")

from typing import List, Iterator, Optional

import torch, os, json, tarfile, argparse, uuid, shutil
import sys

from overrides import overrides

from allennlp.common.util import lazy_groups_of
from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.util import JsonDict, sanitize
from allennlp.common.util import get_spacy_model
from allennlp.data import Instance
from allennlp.data.dataset import Batch
from allennlp.data.fields import ListField, SpanField
from allennlp.nn.util import move_to_device
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import JsonDict, Predictor

from allennlp.common.file_utils import cached_path
from qasrl.data.util import read_lines, get_verb_fields

from qasrl.data.dataset_readers import QasrlReader
from qasrl.models.span import SpanModel
from qasrl.models.span_to_question import SpanToQuestionModel
from qasrl.util.archival_utils import load_archive_from_folder

span_minimum_threshold_default = 0.3
question_minimum_threshold_default = 0.1
question_beam_size_default = 10

class AFirstPipeline():
    def __init__(self,
                 span_model: SpanModel,
                 span_model_dataset_reader: QasrlReader,
                 span_to_question_model: SpanToQuestionModel,
                 span_to_question_model_dataset_reader: QasrlReader,
                 span_minimum_threshold: float = span_minimum_threshold_default,
                 question_minimum_threshold: float = question_minimum_threshold_default,
                 question_beam_size: int = question_beam_size_default) -> None:
        self._span_model = span_model
        self._span_model_dataset_reader = span_model_dataset_reader
        self._span_to_question_model = span_to_question_model
        self._span_to_question_model_dataset_reader = span_to_question_model_dataset_reader
        self._span_minimum_threshold = span_minimum_threshold
        self._question_minimum_threshold = question_minimum_threshold
        self._question_beam_size = question_beam_size

    def predict(self, inputs: JsonDict) -> JsonDict:
        # produce different sets of instances to account for
        # the possibility of different token indexers as well as different vocabularies
        span_verb_instances = list(self._span_model_dataset_reader.sentence_json_to_instances(inputs, verbs_only = True))
        span_to_question_verb_instances = list(self._span_to_question_model_dataset_reader.sentence_json_to_instances(inputs, verbs_only = True))
        span_outputs = self._span_model.forward_on_instances(span_verb_instances)
        verb_dicts = []
        for (verb_instance, span_output) in zip(span_to_question_verb_instances, span_outputs):
            beam = []
            scored_spans = [(s, p) for s, p in span_output["spans"] if p >= self._span_minimum_threshold]
            span_fields = [SpanField(span.start(), span.end(), verb_instance["text"]) for span, _ in scored_spans]
            if len(span_fields) > 0:
                verb_instance.index_fields(self._span_to_question_model.vocab)
                verb_instance.add_field("answer_spans", ListField(span_fields), self._span_to_question_model.vocab)
                qgen_input_tensors = move_to_device(
                    Batch([verb_instance]).as_tensor_dict(),
                    self._span_to_question_model._get_prediction_device())
                question_beams = self._span_to_question_model.beam_decode(
                    text = qgen_input_tensors["text"],
                    predicate_indicator = qgen_input_tensors["predicate_indicator"],
                    predicate_index = qgen_input_tensors["predicate_index"],
                    answer_spans = qgen_input_tensors["answer_spans"],
                    max_beam_size = self._question_beam_size,
                    min_beam_probability = self._question_minimum_threshold)
                for (span, span_prob), (_, slot_values, question_probs) in zip(scored_spans, question_beams):
                    for i in range(len(question_probs)):
                        question_slots = {
                            slot_name: slot_values[slot_name][i]
                            for slot_name in self._span_to_question_model.get_slot_names()
                        }
                        beam.append({
                            "questionSlots": question_slots,
                            "questionProb": question_probs[i],
                            "span": [span.start(), span.end() + 1],
                            "spanProb": span_prob
                        })
            verb_dicts.append({
                "verbIndex": verb_instance["metadata"]["verb_index"],
                "verbInflectedForms": verb_instance["metadata"]["verb_inflected_forms"],
                "beam": beam
            })
        return {
            "sentenceId": inputs["sentenceId"],
            "sentenceTokens": inputs["sentenceTokens"],
            "verbs": verb_dicts
        }

def main(span_model_path: str,
         span_to_question_model_path: str,
         cuda_device: int,
         input_file: str,
         output_file: str,
         span_min_prob: float,
         question_min_prob: float,
         question_beam_size: int) -> None:
    check_for_gpu(cuda_device)
    span_model_archive = load_archive(
        span_model_path,
        cuda_device = cuda_device,
        # weights_file = os.path.join(span_model_path, "best.th")
        )
    span_to_question_model_archive = load_archive(
        span_to_question_model_path,
        cuda_device = cuda_device,
        # weights_file = os.path.join(span_to_question_model_path, "best.th")
        )
    pipeline = AFirstPipeline(
        span_model = span_model_archive.model,
        span_model_dataset_reader = DatasetReader.from_params(span_model_archive.config["dataset_reader"].duplicate()),
        span_to_question_model = span_to_question_model_archive.model,
        span_to_question_model_dataset_reader = DatasetReader.from_params(span_to_question_model_archive.config["dataset_reader"].duplicate()),
        span_minimum_threshold = span_min_prob,
        question_minimum_threshold = question_min_prob,
        question_beam_size = question_beam_size)
    if output_file is None:
        for line in read_lines(cached_path(input_file)):
            input_json = json.loads(line)
            output_json = pipeline.predict(input_json)
            print(json.dumps(output_json))
    else:
        with open(output_file, 'w', encoding = 'utf8') as out:
            for line in read_lines(cached_path(input_file)):
                input_json = json.loads(line)
                output_json = pipeline.predict(input_json)
                print(json.dumps(output_json), file = out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run the answer-first pipeline")
    parser.add_argument('--span', type=str, help = "Path to span detector model archive (.tar.gz).")
    parser.add_argument('--span_to_question', type=str, help = "Path to span-to-question generator model archive (.tar.gz).")
    parser.add_argument('--cuda_device', type=int, default=-1)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str, default = None)
    parser.add_argument('--span_min_prob', type=float, default = span_minimum_threshold_default)
    parser.add_argument('--question_min_prob', type=float, default = question_minimum_threshold_default)
    parser.add_argument('--question_beam_size', type=int, default = question_beam_size_default)

    args = parser.parse_args()
    main(span_model_path = args.span,
         span_to_question_model_path = args.span_to_question,
         cuda_device = args.cuda_device,
         input_file = args.input_file,
         output_file = args.output_file,
         span_min_prob = args.span_min_prob,
         question_min_prob = args.question_min_prob,
         question_beam_size = args.question_beam_size)
