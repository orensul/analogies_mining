from typing import List

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.common.util import get_spacy_model
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor

from qasrl.data.dataset_readers import QasrlReader
from qasrl.data.util import get_verb_fields

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

@Predictor.register("qasrl_clause_answering")
class ClauseAnsweringPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: QasrlReader) -> None:
        super(ClauseAnsweringPredictor, self).__init__(model, dataset_reader)

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        results = self.predict_batch_json([inputs])
        if len(results) > 0:
            return results[0]
        else:
            return {}

    @overrides
    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        sentence_ids = [s["sentenceId"] for s in inputs]
        input_metas = []
        input_instances = []
        for sentence_json in inputs:
            sentence_id = sentence_json["sentenceId"]
            sentence_tokens = sentence_json["sentenceTokens"]
            for verb_json in sentence_json["verbs"]:
                verb_index = int(verb_json["verbIndex"])
                verb_dict = get_verb_fields(self._dataset_reader._token_indexers, sentence_tokens, verb_index)
                for clause_info in verb_json["clauses"]:
                    clause_field = LabelField(label = clause_info["clause"], label_namespace = "clause-template-labels")
                    answer_slot_field = LabelField(label = clause_info["slot"], label_namespace = "answer-slot-labels")
                    input_metas.append({
                        "sentenceId": sentence_id,
                        "verbIndex": verb_index,
                        "clauseInfo": clause_info
                    })
                    input_instances.append(Instance({
                        **verb_dict,
                        "clause": clause_field,
                        "answer_slot": answer_slot_field
                    }))
        if len(input_instances) == 0:
            return []
        # make sure to reproduce desired batch size
        outputs = []
        for input_batch in chunks(input_instances, len(inputs)):
            outputs.extend(sanitize(self._model.forward_on_instances(input_batch)))
        outputs_grouped = { sid: {} for sid in sentence_ids}
        def get(targ, key, default):
            if key not in targ:
                targ[key] = default
            return targ[key]
        for input_meta, output in zip(input_metas, outputs):
            sentence = outputs_grouped[input_meta["sentenceId"]]
            verb = get(sentence, str(input_meta["verbIndex"]), [])
            spans = [[[s[0], s[1]], p] for s, p in output["spans"]]
            results = {
                "question": input_meta["clauseInfo"],
                "spans": output["spans"]
            }
            verb.append(results)
        return [{ "sentenceId": sid, "verbs": outputs_grouped[sid] } for sid in sentence_ids]




