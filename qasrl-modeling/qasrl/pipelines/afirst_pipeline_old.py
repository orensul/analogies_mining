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

span_minimum_threshold_default = 0.05

from nrl.models.qasrl_parser import QaSrlParser
from nrl.data.dataset_readers.qasrl_reader import QaSrlReader

class AFirstPipelineOld():
    def __init__(self,
                 model: QaSrlParser,
                 dataset_reader: QaSrlReader,
                 span_minimum_threshold: float = span_minimum_threshold_default) -> None:
        self._model = model
        self._dataset_reader = dataset_reader
        self._span_minimum_threshold = span_minimum_threshold

    def predict(self, inputs: JsonDict) -> JsonDict:
        verb_instances = []
        verb_entries = []
        for verb_index, verb_entry in inputs["verbEntries"].items():
            verb_entries.append(verb_entry)
            verb_instances.append(self._dataset_reader._make_instance_from_text(inputs["sentenceTokens"], int(verb_index)))
        span_outputs = self._model.span_detector.forward_on_instances(verb_instances)

        verb_dicts = []
        for instance, span_output, verb_entry in zip(verb_instances, span_outputs, verb_entries):
            text_field = instance["text"]
            scored_spans = [(s, p.item()) for s, p in span_output['spans'] if p >= self._span_minimum_threshold]
            beam = []
            if len(scored_spans) > 0:
                labeled_span_field = ListField([SpanField(span.start(), span.end(), text_field) for span, _ in scored_spans])
                instance.add_field("labeled_spans", labeled_span_field, self._model.vocab)
                qg_output = self._model.question_predictor.forward_on_instance(instance)
                for question, (span, span_prob) in zip(qg_output["questions"], scored_spans):
                    s = list(question)
                    question_slots = {
                        "wh": s[0],
                        "aux": s[1],
                        "subj": s[2],
                        "verb": s[3],
                        "obj": s[4],
                        "prep": s[5],
                        "obj2": s[6]
                    }
                    beam.append({
                        "questionSlots": question_slots,
                        "questionProb": 1.0, # TODO add probabilities to output later, maybe.
                        "span": [span.start(), span.end() + 1],
                        "spanProb": span_prob
                    })
            verb_dicts.append({
                "verbIndex": verb_entry["verbIndex"],
                "verbInflectedForms": verb_entry["verbInflectedForms"],
                "beam": beam
            })
        return {
            "sentenceId": inputs["sentenceId"],
            "sentenceTokens": inputs["sentenceTokens"],
            "verbs": verb_dicts
        }

def main(model_path: str,
         cuda_device: int,
         input_file: str,
         output_file: str,
         span_min_prob: float) -> None:
    check_for_gpu(cuda_device)
    model_archive = load_archive(model_path, cuda_device = cuda_device)
    model_archive.model.eval()
    pipeline = AFirstPipelineOld(
        model = model_archive.model,
        dataset_reader = DatasetReader.from_params(model_archive.config["dataset_reader"].duplicate()),
        span_minimum_threshold = span_min_prob)
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
    parser = argparse.ArgumentParser(description = "Run the old answer-first pipeline")
    parser.add_argument('--model', type=str, help = "Path to model archive (.tar.gz).")
    parser.add_argument('--cuda_device', type=int, default=-1)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str, default = None)
    parser.add_argument('--span_min_prob', type=float, default = span_minimum_threshold_default)

    args = parser.parse_args()
    main(model_path = args.model,
         cuda_device = args.cuda_device,
         input_file = args.input_file,
         output_file = args.output_file,
         span_min_prob = args.span_min_prob)
