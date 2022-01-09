from typing import NamedTuple, Dict, List
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.fields import Field, IndexField, TextField, SequenceLabelField, LabelField, ListField, MetadataField, SpanField
from allennlp.data.tokenizers import Token
import codecs
import gzip

from collections import Counter

from qasrl.common.span import Span

def cleanse_sentence_text(sent_text):
    sent_text = ["?" if w == "/?" else w for w in sent_text]
    sent_text = ["." if w == "/." else w for w in sent_text]
    sent_text = ["-" if w == "/-" else w for w in sent_text]
    sent_text = ["(" if w == "-LRB-" else w for w in sent_text]
    sent_text = [")" if w == "-RRB-" else w for w in sent_text]
    sent_text = ["[" if w == "-LSB-" else w for w in sent_text]
    sent_text = ["]" if w == "-RSB-" else w for w in sent_text]
    return sent_text


# used for normal slots and abstract slots
def get_slot_label_namespace(slot_name: str) -> str:
    return "slot_%s_labels" % slot_name

def read_lines(file_path):
    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'r') as f:
            for line in f:
                yield line
    else:
        with codecs.open(file_path, 'r', encoding='utf8') as f:
            for line in f:
                yield line

def read_clause_info(target, file_path):
    def get(targ, key):
        if key not in targ:
            targ[key] = {}
        return targ[key]
    def add_json(obj):
        sentence = get(target, obj["sentenceId"])
        verb = get(sentence, obj["verbIndex"])
        clause = get(verb, obj["question"])
        slot_dict = obj["slots"]
        slot_dict["qarg"] = obj["answerSlot"]
        clause["slots"] = slot_dict

    import json
    for line in read_lines(file_path):
        add_json(json.loads(line))
    return

def read_simple_clause_info(target, file_path):
    import json
    for line in read_lines(file_path):
        obj = json.loads(line)
        target[obj["sentenceId"]] = obj["verbs"]
    return

qasrl_slot_names = ["wh", "aux", "subj", "verb", "obj", "prep", "obj2"]

abstract_slot_names = [
    ("wh", lambda x: "what" if (x == "who") else x),
    ("subj", lambda x: "something" if (x == "someone") else x),
    ("obj", lambda x: "something" if (x == "someone") else x),
    ("obj2", lambda x: "something" if (x == "someone") else x),
    ("prep", lambda x: "_" if (x == "_") else "<prep>")
]

### Field constructors

def get_verb_fields(token_indexers: Dict[str, TokenIndexer],
                    sentence_tokens: List[str],
                    verb_index: int):
    text_field = TextField([Token(t) for t in sentence_tokens], token_indexers)
    return {
        "text": text_field,
        "predicate_index": IndexField(verb_index, text_field),
        "predicate_indicator": SequenceLabelField(
            [1 if i == verb_index else 0 for i in range(len(sentence_tokens))], text_field)
    }

def get_question_slot_fields(question_slots):
    """
    Input: json dicty thing from QA-SRL Bank
    Output: dict from question slots to fields
    """
    def get_slot_value_field(slot_name):
        slot_value = question_slots[slot_name]
        namespace = get_slot_label_namespace(slot_name)
        return LabelField(label = slot_value, label_namespace = namespace)
    return { slot_name : get_slot_value_field(slot_name) for slot_name in qasrl_slot_names }

# def get_abstract_question_slots(question_label):
#     def get_abstract_slot_value(slot_name, get_abstracted_value):
#         return get_abstracted_value(question_label["questionSlots"][slot_name])
#     abstract_slots_dict = { ("abst-%s" % slot_name): get_abstract_slot_value(slot_name, get_abstracted_value)
#                                    for slot_name, get_abstracted_value in abstract_slot_names }
#     abst_verb_value = "verb[pss]" if question_label["isPassive"] else "verb"
#     return {**abstract_slots_dict, **{"abst-verb": abst_verb_value}}

def get_abstract_question_slot_fields(question_label):
    def get_abstract_slot_value_field(slot_name, get_abstracted_value):
        abst_slot_name = "abst-%s" % slot_name
        namespace = get_slot_label_namespace(abst_slot_name)
        abst_slot_value = get_abstracted_value(question_label["questionSlots"][slot_name])
        return LabelField(label = abst_slot_value, label_namespace = namespace)
    direct_abstract_slots_dict = { ("abst-%s" % slot_name): get_abstract_slot_value_field(slot_name, get_abstracted_value)
                                for slot_name, get_abstracted_value in abstract_slot_names }
    abst_verb_value = "verb[pss]" if question_label["isPassive"] else "verb"
    abst_verb_field = LabelField(label = abst_verb_value, label_namespace = get_slot_label_namespace("abst-verb"))
    return {**direct_abstract_slots_dict, **{"abst-verb": abst_verb_field}}

def get_clause_slot_field(slot_name: str, slot_value: str):
    clause_slot_name = "clause-%s" % slot_name
    namespace = get_slot_label_namespace(clause_slot_name)
    return LabelField(label = slot_value, label_namespace = namespace)

def get_answer_spans(question_labels):
    spans = [Span(s[0], s[1]-1) for q in question_labels for ans in q["answerJudgments"] if ans["isValid"] for s in ans["spans"]]
    span_counts = Counter()
    for span in spans:
        span_counts[span] += 1
    distinct_spans = list(span_counts)
    distinct_span_counts = [span_counts[c] for c in distinct_spans]
    return distinct_spans, distinct_span_counts

def get_answer_spans_field(answer_spans, text_field):
    span_list = [SpanField(s.start(), s.end(), text_field) for s in answer_spans]
    if len(span_list) == 0:
        return ListField([SpanField(-1, -1, text_field)])
    else:
        return ListField(span_list)

def get_num_answers_field(question_label):
    return LabelField(label = len(question_label["answerJudgments"]), skip_indexing = True)

def get_num_valids_field(question_label):
    return LabelField(label = len([aj for aj in question_label["answerJudgments"] if aj["isValid"]]), skip_indexing = True)

def get_num_invalids(question_label):
    return len(question_label["answerJudgments"]) - len([aj for aj in question_label["answerJudgments"] if aj["isValid"]])

def get_num_invalids_field(question_label):
    return LabelField(label = get_num_invalids(question_label), skip_indexing = True)

def get_answer_fields(question_label, text_field):
    spans, span_counts = get_answer_spans([question_label])
    if len(span_counts) == 0:
        span_counts_field = ListField([LabelField(label = -1, skip_indexing = True)])
    else:
        span_counts_field = ListField([LabelField(label = count, skip_indexing = True) for count in span_counts])
    answer_spans_field = get_answer_spans_field(spans, text_field)
    num_answers_field = get_num_answers_field(question_label)
    num_valids_field = get_num_valids_field(question_label)
    num_invalids_field = get_num_invalids_field(question_label)
    return {
        "answer_spans": answer_spans_field,
        "span_counts": span_counts_field,
        "num_answers": num_answers_field,
        "num_valids": num_valids_field,
        "num_invalids": num_invalids_field
    }

def get_tan_string(question_label):
    tense_string = question_label["tense"]
    perfect_string = "+pf" if question_label["isPerfect"] else "-pf"
    progressive_string = "+prog" if question_label["isProgressive"] else "-prog"
    negation_string = "+neg" if question_label["isNegated"] else "-neg"
    tan_string = " ".join([tense_string, perfect_string, progressive_string, negation_string])
    return tan_string
