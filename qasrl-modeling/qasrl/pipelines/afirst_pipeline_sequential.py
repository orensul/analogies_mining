 # completely ridiculous hack to import stuff properly. somebody save me from myself
import importlib
from allennlp.common.util import import_submodules
importlib.invalidate_caches()
import sys
sys.path.append("/Users/orensultan/PycharmProjects/TextAnalogiesMining/qasrl-modeling/")
print(sys.path)
import_submodules("qasrl")

from typing import List, Iterator, Optional, Set

import torch, os, json, tarfile, argparse, uuid, shutil
import gzip
import sys

from tqdm import tqdm

from overrides import overrides

from allennlp.common.file_utils import cached_path
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

from qasrl.common.span import Span
from qasrl.data.dataset_readers import QasrlReader
from qasrl.data.util import read_lines, get_verb_fields
from qasrl.models.span import SpanModel
from qasrl.models.span_to_question import SpanToQuestionModel
from qasrl.util.archival_utils import load_archive_from_folder

span_minimum_threshold_default = 0.3
question_minimum_threshold_default = 0.01
question_beam_size_default = 20

# Modifies original pipeline to output a hierarchical representation of the beam (span -> question).
class AFirstPipelineSequential():
    """
    Span -> question pipeline. Outputs a sentence per line as a JSON object.
    The output format for each sentence is:
    {
        "sentenceId": String,
        "sentenceTokens": List[String],
        "verbs": List[{
            "verbIndex": Int,
            "verbInflectedForms": { "stem": String, ... } # same as QA-SRL verb inflected forms
            "beam": Beam
        }]
    }
    where the sentenceId, tokens, and indices of each verb (predicate) are taken from the input
    (which is expected to have these in the QA-SRL Bank 2.0 format). Inflected forms are only
    included in the output if they are found in the input (as they will be when running directly on
    the QA-SRL Bank). The "Beam" object contains the model predictions, and is formatted as follows:
    Beam: List[{
        "span": [Int, Int], # [beginning (inclusive), end (exclusive)] formatted as a 2-element list
        "spanProb": Float, # the probability assigned to the span by the span detection model
        "questions": List[{
            "questionSlots": QuestionSlots, # JSON object with question slots, in QA-SRL Bank format
            "questionProb": Float # probability assigned to the question by the question gen model
        }]
    }]
    The beam will contain an entry for every span that receives a probability above the
    `span_minimum_threshold` argument as well as any spans that were already provided in the input
    JSON. The "questions" field for each span will contain an entry for every question output during
    beam search. The beam has a size cutoff of `question_beam_size` and a probability cutoff of
    `question_minimum_threshold`.
    """
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

    # if there are no spans found in the input, that's fine; just don't add any required ones
    def _get_verb_spans_for_sentence(self, inputs: JsonDict) -> List[Set[Span]]:
        verb_spans = []
        verb_entries = [v for _, v in inputs["verbEntries"].items()]
        verb_entries = sorted(verb_entries, key = lambda v: v["verbIndex"])
        for verb in verb_entries:
            spans = []
            if "questionLabels" in verb:
                for _, question_label in verb["questionLabels"].items():
                    for aj in question_label["answerJudgments"]:
                        if aj["isValid"]:
                            for s in aj["spans"]:
                                spans.append(Span(s[0], s[1] - 1))
            if "argumentSpans" in verb:
                for s in verb["argumentSpans"]:
                    spans.append(Span(s[0], s[1] - 1))
            verb_spans.append(set(spans))
        return verb_spans

    def predict(self, inputs: JsonDict) -> JsonDict:
        # produce different sets of instances to account for
        # the possibility of different token indexers as well as different vocabularies
        span_verb_instances = list(self._span_model_dataset_reader.sentence_json_to_instances(inputs, verbs_only = True))
        span_to_question_verb_instances = list(self._span_to_question_model_dataset_reader.sentence_json_to_instances(inputs, verbs_only = True))
        verb_spans = self._get_verb_spans_for_sentence(inputs)
        # get spans and ensure same order
        verb_dicts = []
        if len(span_verb_instances) > 0:
            span_outputs = self._span_model.forward_on_instances(span_verb_instances)
            for (verb_instance, span_output, ref_spans) in zip(span_to_question_verb_instances, span_outputs, verb_spans):
                beam = []
                scored_spans = [
                    (s, p)
                    for s, p in span_output["spans"]
                    if p >= self._span_minimum_threshold or s in ref_spans # always include reference spans
                ]
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
                        scored_questions = []
                        for i in range(len(question_probs)):
                            question_slots = {
                                slot_name: slot_values[slot_name][i]
                                for slot_name in self._span_to_question_model.get_slot_names()
                            }
                            scored_questions.append({
                                "questionSlots": question_slots,
                                "questionProb": question_probs[i]
                            })
                        beam.append({
                            "span": [span.start(), span.end() + 1],
                            "spanProb": span_prob,
                            "questions": scored_questions
                        })
                verb_entry = {"verbIndex": verb_instance["metadata"]["verb_index"]}
                if "verb_inflected_forms" in verb_instance["metadata"]:
                    verb_entry["verbInflectedForms"] = verb_instance["metadata"]["verb_inflected_forms"]
                verb_entry["beam"] = beam
                verb_dicts.append(verb_entry)
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
        overrides = '{ "model": { "span_selector": {"span_decoding_threshold": 0.00} } }',
        # weights_file = os.path.join(span_model_path, "best.th")
        )

    # override span detection threshold to be low enough so we can reasonably approximate bad spans
    # as having probability 0.
    span_to_question_model_archive = load_archive(
        span_to_question_model_path,
        cuda_device = cuda_device,
        # weights_file = os.path.join(span_to_question_model_path, "best.th")
        )

    span_model_dataset_reader_params = span_model_archive.config["dataset_reader"].duplicate()
    span_model_dataset_reader_params["qasrl_filter"]["allow_all"] = True

    span_to_question_model_dataset_reader_params = span_to_question_model_archive.config["dataset_reader"].duplicate()
    span_to_question_model_dataset_reader_params["qasrl_filter"]["allow_all"] = True

    pipeline = AFirstPipelineSequential(
        span_model = span_model_archive.model,
        span_model_dataset_reader = DatasetReader.from_params(span_model_dataset_reader_params),
        span_to_question_model = span_to_question_model_archive.model,
        span_to_question_model_dataset_reader = DatasetReader.from_params(span_to_question_model_dataset_reader_params),
        span_minimum_threshold = span_min_prob,
        question_minimum_threshold = question_min_prob,
        question_beam_size = question_beam_size)
    if output_file is None:
        for line in tqdm(read_lines(cached_path(input_file))):
            input_json = json.loads(line)
            output_json = pipeline.predict(input_json)
            print(json.dumps(output_json))
    elif output_file.endswith('.gz'):
        with gzip.open(output_file, 'wt') as f:
            for line in tqdm(read_lines(cached_path(input_file))):
                input_json = json.loads(line)
                output_json = pipeline.predict(input_json)
                f.write(json.dumps(output_json))
                f.write('\n')
    else:
        with open(output_file, 'w', encoding = 'utf8') as out:
            for line in tqdm(read_lines(cached_path(input_file))):
                input_json = json.loads(line)
                output_json = pipeline.predict(input_json)
                print(json.dumps(output_json), file = out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run the answer-first pipeline")
    parser.add_argument('--span', type=str, help = "Path to span detector model serialization dir.")
    parser.add_argument('--span_to_question', type=str, help = "Path to span-to-question generator serialization dir.")
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
