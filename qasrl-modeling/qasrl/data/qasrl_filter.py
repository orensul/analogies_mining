from typing import List
from allennlp.common import Registrable

from qasrl.data.util import cleanse_sentence_text

class QasrlFilter(Registrable):
    def __init__(self,
                 min_answers: int = 1,
                 min_valid_answers: int = 0,
                 domains: List[str] = None,
                 question_sources: List[str] = None,
                 allow_all: bool = False):
        self._min_answers = min_answers
        self._min_valid_answers = min_valid_answers
        self._domains = [d.lower() for d in domains] if domains is not None else None
        self._question_sources = question_sources
        self._allow_all = allow_all
    def filter_sentence(self, sentence_json): # -> Iterable[Dict[str, ?]]
        is_sentence_in_domain = self._domains is None or any([d in sentence_json["sentenceId"].lower() for d in self._domains])
        base_dict = {
            "sentence_id": sentence_json["sentenceId"],
            "sentence_tokens": cleanse_sentence_text(sentence_json["sentenceTokens"]),
        }
        if is_sentence_in_domain or self._allow_all:
            verb_entries = [v for _, v in sentence_json["verbEntries"].items()]
            verb_entries = sorted(verb_entries, key = lambda v: v["verbIndex"])
            for verb_entry in verb_entries:
                verb_dict = {
                    "verb_index": verb_entry["verbIndex"]
                }

                if "questionLabels" in verb_entry:
                    def is_valid(question_label):
                        if self._allow_all:
                            return True
                        answers = question_label["answerJudgments"]
                        valid_answers = [a for a in answers if a["isValid"]]
                        is_source_valid = self._question_sources is None or any([l.startswith(source) for source in self._question_sources for l in question_label["questionSources"]])
                        return (len(answers) >= self._min_answers) and (len(valid_answers) >= self._min_valid_answers) and is_source_valid
                    question_labels = [l for q, l in verb_entry["questionLabels"].items() if is_valid(l)]
                    verb_dict["question_labels"] = question_labels

                if "verbInflectedForms" in verb_entry:
                    verb_dict["verb_inflected_forms"] = verb_entry["verbInflectedForms"]

                yield {**base_dict, **verb_dict}
