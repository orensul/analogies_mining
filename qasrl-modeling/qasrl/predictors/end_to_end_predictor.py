from typing import List

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.common.util import get_spacy_model
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor

from qasrl.data.dataset_readers import QasrlReader

@Predictor.register("qasrl_end_to_end")
class EndToEndPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: QasrlReader) -> None:
        super(EndToEndPredictor, self).__init__(model, dataset_reader)

    @overrides
    def predict_json(self, inputs: JsonDict) -> List[JsonDict]:
        instances = list(self._dataset_reader.sentence_json_to_instances(inputs, verbs_only = True))
        results = sanitize(self._model.forward_on_instances(instances))
        def get_verb_dict(instance, result):
            return {
                "verbIndex": instance["metadata"]["verb_index"],
                "verbInflectedForms": instance["metadata"]["verb_inflected_forms"],
                "animacies": result["animacies"],
                "tans": result["tans"],
                "beam": result["beam"]
            }
        result_dict = {
            "sentenceId": inputs["sentenceId"],
            "sentenceTokens": inputs["sentenceTokens"],
            "verbs": [get_verb_dict(i, r) for i, r in zip(instances, results)]
        }
        return result_dict

    @overrides
    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        return map(inputs, self.predict_json)
