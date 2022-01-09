 # completely ridiculous hack to import stuff properly. somebody save me from myself
import importlib
from allennlp.common.util import import_submodules
importlib.invalidate_caches()
import sys
sys.path.append(".")
import_submodules("qasrl")

from typing import List, Iterator, Optional, Dict

import torch, os, json, tarfile, argparse, uuid, shutil
import sys

from overrides import overrides

from allennlp.common.util import lazy_groups_of
from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.util import JsonDict, sanitize
from allennlp.common.util import get_spacy_model
from allennlp.data import Instance
from allennlp.data.dataset import Batch
from allennlp.data.fields import ListField, SpanField, LabelField
from allennlp.nn.util import move_to_device
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import JsonDict, Predictor

from allennlp.common.file_utils import cached_path
from qasrl.data.util import read_lines, get_verb_fields

from qasrl.data.dataset_readers import QasrlReader
from qasrl.models.multiclass import MulticlassModel
from qasrl.models.span import SpanModel
from qasrl.models.span_to_tan import SpanToTanModel
from qasrl.models.animacy import AnimacyModel
from qasrl.models.clause_and_span_to_answer_slot import ClauseAndSpanToAnswerSlotModel
from qasrl.util.archival_utils import load_archive_from_folder

clause_minimum_threshold_default = 0.10
span_minimum_threshold_default = 0.10
tan_minimum_threshold_default = 0.20

class FactoredPipeline():
    def __init__(self,
                 span_model: SpanModel,
                 span_model_dataset_reader: QasrlReader,
                 clause_model: MulticlassModel,
                 clause_model_dataset_reader: QasrlReader,
                 answer_slot_model: ClauseAndSpanToAnswerSlotModel,
                 answer_slot_model_dataset_reader: QasrlReader,
                 tan_model: MulticlassModel,
                 tan_model_dataset_reader: QasrlReader,
                 span_to_tan_model: SpanToTanModel,
                 span_to_tan_model_dataset_reader: QasrlReader,
                 animacy_model: AnimacyModel,
                 animacy_model_dataset_reader: QasrlReader,
                 clause_minimum_threshold: float = span_minimum_threshold_default,
                 span_minimum_threshold: float = clause_minimum_threshold_default,
                 tan_minimum_threshold: float = tan_minimum_threshold_default) -> None:
        self._span_model = span_model
        self._span_model_dataset_reader = span_model_dataset_reader
        self._clause_model = clause_model
        self._clause_model_dataset_reader = clause_model_dataset_reader
        self._answer_slot_model = answer_slot_model
        self._answer_slot_model_dataset_reader = answer_slot_model_dataset_reader
        self._tan_model = tan_model
        self._tan_model_dataset_reader = tan_model_dataset_reader
        self._span_to_tan_model = span_to_tan_model
        self._span_to_tan_model_dataset_reader = span_to_tan_model_dataset_reader
        self._animacy_model = animacy_model
        self._animacy_model_dataset_reader = animacy_model_dataset_reader
        self._clause_minimum_threshold = clause_minimum_threshold
        self._span_minimum_threshold = span_minimum_threshold
        self._tan_minimum_threshold = tan_minimum_threshold

    def predict(self, inputs: JsonDict) -> JsonDict:
        clause_instances = list(self._clause_model_dataset_reader.sentence_json_to_instances(inputs, verbs_only = True))
        clause_outputs = self._clause_model.forward_on_instances(clause_instances)
        span_instances = list(self._span_model_dataset_reader.sentence_json_to_instances(inputs, verbs_only = True))
        span_outputs = self._span_model.forward_on_instances(span_instances)
        tan_instances = list(self._tan_model_dataset_reader.sentence_json_to_instances(inputs, verbs_only = True))
        tan_outputs = self._tan_model.forward_on_instances(tan_instances)

        answer_slot_instances = self._answer_slot_model_dataset_reader.sentence_json_to_instances(inputs, verbs_only = True)
        span_to_tan_instances = list(self._span_to_tan_model_dataset_reader.sentence_json_to_instances(inputs, verbs_only = True))
        animacy_instances = list(self._animacy_model_dataset_reader.sentence_json_to_instances(inputs, verbs_only = True))

        verb_dicts = []
        for answer_slot_instance, clause_output, span_output, tan_output, span_to_tan_instance, animacy_instance in zip(answer_slot_instances, clause_outputs, span_outputs, tan_outputs, span_to_tan_instances, animacy_instances):
            beam = []
            scored_spans = [
                (s, p)
                for s, p in span_output["spans"]
                if p >= self._span_minimum_threshold]
            all_spans = [s for s, _ in scored_spans]
            if len(all_spans) > 0:
                span_to_tan_instance.add_field("tan_spans", ListField([SpanField(s.start(), s.end(), span_to_tan_instance["text"]) for s in all_spans]))
                span_to_tan_output = self._span_to_tan_model.forward_on_instance(span_to_tan_instance)
                animacy_instance.add_field("animacy_spans", ListField([SpanField(s.start(), s.end(), animacy_instance["text"]) for s in all_spans]))
                animacy_output = self._animacy_model.forward_on_instance(animacy_instance)
            else:
                span_to_tan_output = None
                animacy_output = None

            scored_clauses = [
                (self._clause_model.vocab.get_token_from_index(c, namespace = "abst-clause-labels"), p)
                for c, p in enumerate(clause_output["probs"].tolist())
                if p >= self._clause_minimum_threshold]
            clause_span_pairs = [
                (clause, clause_prob, span, span_prob)
                for clause, clause_prob in scored_clauses
                for span, span_prob in scored_spans
            ]
            answer_slot_clause_fields = [
                LabelField(clause, label_namespace = "abst-clause-labels")
                for clause, _, _, _ in clause_span_pairs
            ]
            answer_slot_span_fields = [
                SpanField(span.start(), span.end(), answer_slot_instance["text"])
                for _, _, span, _ in clause_span_pairs
            ]
            qa_beam = []
            if len(answer_slot_clause_fields) > 0:
                answer_slot_instance.add_field("qarg_labeled_clauses", ListField(answer_slot_clause_fields))
                answer_slot_instance.add_field("qarg_labeled_spans", ListField(answer_slot_span_fields))
                answer_slot_output = self._answer_slot_model.forward_on_instance(answer_slot_instance)

                for i in range(len(clause_span_pairs)):
                    clause, clause_prob, span, span_prob = clause_span_pairs[i]
                    answer_slots_with_probs = {
                        self._answer_slot_model.vocab.get_token_from_index(slot_index, namespace = "qarg-labels"): answer_slot_output["probs"][i, slot_index].item()
                        for slot_index in range(self._answer_slot_model.vocab.get_vocab_size("qarg-labels"))
                    }
                    qa_beam.append({
                        "clause": clause,
                        "clauseProb": clause_prob,
                        "span": [span.start(), span.end() + 1],
                        "spanProb": span_prob,
                        "answerSlots": answer_slots_with_probs
                    })
            beam = {
                "qa_beam": qa_beam,
                "tans": [
                    (self._tan_model.vocab.get_token_from_index(i, namespace = "tan-string-labels"), p)
                    for i, p in enumerate(tan_output["probs"].tolist())
                    if p >= self._tan_minimum_threshold
                ]
            }
            if span_to_tan_output is not None:
                beam["span_tans"] = [
                    ([s.start(), s.end() + 1], [
                        (self._span_to_tan_model.vocab.get_token_from_index(i, namespace = "tan-string-labels"), p)
                        for i, p in enumerate(probs)
                        if p >= self._tan_minimum_threshold])
                    for s, probs in zip(all_spans, span_to_tan_output["probs"].tolist())
                ]
            if animacy_output is not None:
                beam["animacy"] = [
                    ([s.start(), s.end() + 1], p)
                    for s, p in zip(all_spans, animacy_output["probs"].tolist())
                ]
            verb_dicts.append({
                "verbIndex": answer_slot_instance["metadata"]["verb_index"],
                "verbInflectedForms": answer_slot_instance["metadata"]["verb_inflected_forms"],
                "beam": beam
            })
        return {
            "sentenceId": inputs["sentenceId"],
            "sentenceTokens": inputs["sentenceTokens"],
            "verbs": verb_dicts
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run the answer-first pipeline")
    parser.add_argument('--clause', type=str, help = "Path to clause detector model serialization dir.")
    parser.add_argument('--span', type=str, help = "Path to span detector model serialization dir.")
    parser.add_argument('--answer_slot', type=str, help = "Path to answer slot model serialization dir.")
    parser.add_argument('--tan', type=str, help = "Path to TAN model serialization dir.")
    parser.add_argument('--span_to_tan', type=str, help = "Path to TAN model serialization dir.")
    parser.add_argument('--animacy', type=str, help = "Path to animacy model serialization dir .")
    parser.add_argument('--cuda_device', type=int, default=-1)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str, default = None)
    parser.add_argument('--clause_min_prob', type=float, default = clause_minimum_threshold_default)
    parser.add_argument('--span_min_prob', type=float, default = span_minimum_threshold_default)
    parser.add_argument('--tan_min_prob', type=float, default = tan_minimum_threshold_default)
    args = parser.parse_args()

    check_for_gpu(args.cuda_device)
    clause_model_archive = load_archive_from_folder(args.clause, cuda_device = args.cuda_device, weights_file = os.path.join(args.clause, "best.th"))
    span_model_archive = load_archive_from_folder(args.span, cuda_device = args.cuda_device, weights_file = os.path.join(args.span, "best.th"))
    answer_slot_model_archive = load_archive_from_folder(args.answer_slot, cuda_device = args.cuda_device, weights_file = os.path.join(args.answer_slot, "best.th"))
    tan_model_archive = load_archive_from_folder(args.tan, cuda_device = args.cuda_device, weights_file = os.path.join(args.tan, "best.th"))
    span_to_tan_model_archive = load_archive_from_folder(args.span_to_tan, cuda_device = args.cuda_device, weights_file = os.path.join(args.span_to_tan, "best.th"))
    animacy_model_archive = load_archive_from_folder(args.animacy, cuda_device = args.cuda_device, weights_file = os.path.join(args.animacy, "best.th"))
    pipeline = FactoredPipeline(
        clause_model = clause_model_archive.model,
        clause_model_dataset_reader = DatasetReader.from_params(clause_model_archive.config["dataset_reader"].duplicate()),
        span_model = span_model_archive.model,
        span_model_dataset_reader = DatasetReader.from_params(span_model_archive.config["dataset_reader"].duplicate()),
        answer_slot_model = answer_slot_model_archive.model,
        answer_slot_model_dataset_reader = DatasetReader.from_params(answer_slot_model_archive.config["dataset_reader"].duplicate()),
        tan_model = tan_model_archive.model,
        tan_model_dataset_reader = DatasetReader.from_params(tan_model_archive.config["dataset_reader"].duplicate()),
        span_to_tan_model = span_to_tan_model_archive.model,
        span_to_tan_model_dataset_reader = DatasetReader.from_params(span_to_tan_model_archive.config["dataset_reader"].duplicate()),
        animacy_model = animacy_model_archive.model,
        animacy_model_dataset_reader = DatasetReader.from_params(animacy_model_archive.config["dataset_reader"].duplicate()),
        clause_minimum_threshold = args.clause_min_prob,
        span_minimum_threshold = args.span_min_prob,
        tan_minimum_threshold = args.tan_min_prob)
    if args.output_file is None:
        for line in read_lines(cached_path(args.input_file)):
            input_json = json.loads(line)
            output_json = pipeline.predict(input_json)
            print(json.dumps(output_json))
    else:
        with open(args.output_file, 'w', encoding = 'utf8') as out:
            for line in read_lines(cached_path(args.input_file)):
                input_json = json.loads(line)
                output_json = pipeline.predict(input_json)
                print(json.dumps(output_json), file = out)
