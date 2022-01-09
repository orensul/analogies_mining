from typing import List, Dict

from allennlp.common import Registrable
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.fields import Field, IndexField, TextField, SequenceLabelField, LabelField, ListField, MetadataField, SpanField
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

from qasrl.common.span import Span
from qasrl.data.fields.multilabel_field_new import MultiLabelField_New
from qasrl.data.fields.number_field import NumberField
from qasrl.data.fields.multiset_field import MultisetField
from qasrl.data.util import *

from overrides import overrides
import random

import logging
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class QasrlInstanceReader(Registrable):
    def read_instances(self,
                       token_indexers: Dict[str, TokenIndexer],
                       sentence_id: str,
                       sentence_tokens: List[str],
                       verb_index: int,
                       verb_inflected_forms: Dict[str, str],
                       question_labels): # Iterable[Dict[str, ?Field]]
        raise NotImplementedError

@QasrlInstanceReader.register("verb_only")
class QasrlVerbOnlyReader(QasrlInstanceReader):
    @overrides
    def read_instances(self,
                       token_indexers: Dict[str, TokenIndexer],
                       sentence_id: str,
                       sentence_tokens: List[str],
                       verb_index: int,
                       verb_inflected_forms: Dict[str, str],
                       question_labels): # Iterable[Instance]
        yield get_verb_fields(token_indexers, sentence_tokens, verb_index)

# simple_clause_arg_slots = ["subj", "obj", "obj2"]

@QasrlInstanceReader.register("clause_answers")
class QasrlClauseAnswersReader(QasrlInstanceReader):
    def __init__(self,
                 clause_info_files: List[str] = []):
        self._clause_info = None
        if len(clause_info_files) > 0:
            self._clause_info = {}
            for file_path in clause_info_files:
                read_simple_clause_info(self._clause_info, file_path)
    @overrides
    def read_instances(self,
                       token_indexers: Dict[str, TokenIndexer],
                       sentence_id: str,
                       sentence_tokens: List[str],
                       verb_index: int,
                       verb_inflected_forms: Dict[str, str],
                       question_labels): # -> Iterable[Instance]
        verb_dict = get_verb_fields(token_indexers, sentence_tokens, verb_index)
        for question_label in question_labels:
            clause_info = self._clause_info[sentence_id][str(verb_index)][question_label["questionString"]]
            clause_field = LabelField(label = clause_info["clause"], label_namespace = "clause-template-labels")
            answer_slot_field = LabelField(label = clause_info["slot"], label_namespace = "answer-slot-labels")
            answer_spans, span_counts = get_answer_spans([question_label])
            answer_spans_field = get_answer_spans_field(answer_spans, verb_dict["text"])
            if len(span_counts) == 0:
                span_counts_field = ListField([NumberField(0.0)])
            else:
                span_counts_field = ListField([NumberField(float(count)) for count in span_counts])
            num_answers = len([aj for aj in question_label["answerJudgments"]])
            num_answers_field = NumberField(num_answers)
            yield {
                **verb_dict,
                "clause": clause_field,
                "answer_slot": answer_slot_field,
                "answer_spans": answer_spans_field,
                "span_counts": span_counts_field,
                "num_answers": num_answers_field,
                "metadata": {"gold_spans": set(answer_spans)}
            }

@QasrlInstanceReader.register("verb_answers")
class QasrlVerbAnswersReader(QasrlInstanceReader):
    @overrides
    def read_instances(self,
                       token_indexers: Dict[str, TokenIndexer],
                       sentence_id: str,
                       sentence_tokens: List[str],
                       verb_index: int,
                       verb_inflected_forms: Dict[str, str],
                       question_labels): # -> Iterable[Instance]
        verb_dict = get_verb_fields(token_indexers, sentence_tokens, verb_index)
        answer_spans, span_counts = get_answer_spans(question_labels)
        answer_spans_field = get_answer_spans_field(answer_spans, verb_dict["text"])
        #if len(span_counts) == 0:
        #    span_counts_field = ListField([NumberField(0.0)])
        #else:
        #    span_counts_field = ListField([NumberField(float(count)) for count in span_counts])

        if len(span_counts) == 0:
            span_counts_field = ListField([LabelField(label = -1, skip_indexing = True)])
        else:
            span_counts_field = ListField([LabelField(label = count, skip_indexing = True) for count in span_counts])

        num_questions = len(question_labels)
        if num_questions > 0:
            num_answers = len([aj for ql in question_labels for aj in ql["answerJudgments"]]) / num_questions
            num_answers_field = NumberField(num_answers)
            yield {
                **verb_dict,
                "answer_spans": answer_spans_field,
                "span_counts": span_counts_field,
                "num_answers": num_answers_field,
                "metadata": {"gold_spans": set(answer_spans)}
            }

@QasrlInstanceReader.register("verb_qas")
class QasrlVerbQAsReader(QasrlInstanceReader):
    def __init__(self,
                 slot_names: List[str] = ["wh", "aux", "subj", "verb", "obj", "prep", "obj2"],
                 clause_info_files: List[str] = []):
        self._slot_names = slot_names
        self._clause_info = None
        if len(clause_info_files) > 0:
            self._clause_info = {}
            for file_path in clause_info_files:
                read_clause_info(self._clause_info, file_path)
        self._tokenizer = WordTokenizer()
    @overrides
    def read_instances(self,
                       token_indexers: Dict[str, TokenIndexer],
                       sentence_id: str,
                       sentence_tokens: List[str],
                       verb_index: int,
                       verb_inflected_forms: Dict[str, str],
                       question_labels): # -> Iterable[Instance]
        verb_dict = get_verb_fields(token_indexers, sentence_tokens, verb_index)
        question_slot_field_lists = {}
        answer_spans = []
        for question_label in question_labels:
            question_tokens = self._tokenizer.tokenize(question_label["questionString"])
            question_text_field = TextField(question_tokens, token_indexers)
            question_slots_dict = get_question_slot_fields(question_label["questionSlots"])
            abstract_slots_dict = get_abstract_question_slot_fields(question_label)
            if self._clause_info is not None and any([s.startswith("clause") for s in self._slot_names]):
                try:
                    clause_slots = self._clause_info[sentence_id][verb_index][question_label["questionString"]]["slots"]
                    def abst_noun(x):
                        return "something" if (x == "someone") else x
                    clause_slots["abst-subj"] = abst_noun(clause_slots["subj"])
                    clause_slots["abst-verb"] = "verb[pss]" if question_label["isPassive"] else "verb"
                    clause_slots["abst-obj"] = abst_noun(clause_slots["obj"])
                    clause_slots["abst-prep1-obj"] = abst_noun(clause_slots["prep1-obj"])
                    clause_slots["abst-prep2-obj"] = abst_noun(clause_slots["prep2-obj"])
                    clause_slots["abst-misc"] = abst_noun(clause_slots["misc"])
                    clause_slots_dict = { ("clause-%s" % k) : get_clause_slot_field(k, v) for k, v in clause_slots.items() }
                except KeyError:
                    logger.info("Omitting instance without clause data: %s / %s / %s" % (sentence_id, verb_index, question_label["questionString"]))
                    continue
            else:
                clause_slots_dict = {}

            all_slots_dict = {**question_slots_dict, **abstract_slots_dict, **clause_slots_dict}
            included_slot_fields = { k: v for k, v in all_slots_dict.items() if k in self._slot_names }
            new_answer_spans, _ = get_answer_spans([question_label])
            for span in new_answer_spans:
                answer_spans.append(span)
                for k, v in included_slot_fields.items():
                    if k not in question_slot_field_lists:
                        question_slot_field_lists[k] = []
                    question_slot_field_lists[k].append(v)

        question_slot_list_fields = {
            k : ListField(v) for k, v in question_slot_field_lists.items()
        }

        if len(answer_spans) > 0:
            yield {
                **verb_dict,
                **question_slot_list_fields,
                "answer_spans": get_answer_spans_field(answer_spans, verb_dict["text"])
            }

@QasrlInstanceReader.register("question")
class QasrlQuestionReader(QasrlInstanceReader):
    def __init__(self,
                 slot_names: List[str],
                 clause_info_files: List[str] = []):
        self._slot_names = slot_names
        self._clause_info = None
        if len(clause_info_files) > 0:
            self._clause_info = {}
            for file_path in clause_info_files:
                read_clause_info(self._clause_info, file_path)
        self._tokenizer = WordTokenizer()
    @overrides
    def read_instances(self,
                       token_indexers: Dict[str, TokenIndexer],
                       sentence_id: str,
                       sentence_tokens: List[str],
                       verb_index: int,
                       verb_inflected_forms: Dict[str, str],
                       question_labels): # Iterable[Dict[str, ?Field]]
        verb_fields = get_verb_fields(token_indexers, sentence_tokens, verb_index)
        for question_label in question_labels:
            question_tokens = self._tokenizer.tokenize(question_label["questionString"])
            question_text_field = TextField(question_tokens, token_indexers)
            question_slots_dict = get_question_slot_fields(question_label["questionSlots"])
            abstract_slots_dict = get_abstract_question_slot_fields(question_label)
            if self._clause_info is not None and any([s.startswith("clause") for s in self._slot_names]):
                try:
                    clause_slots = self._clause_info[sentence_id][verb_index][question_label["questionString"]]["slots"]
                    def abst_noun(x):
                        return "something" if (x == "someone") else x
                    clause_slots["abst-subj"] = abst_noun(clause_slots["subj"])
                    clause_slots["abst-verb"] = "verb[pss]" if question_label["isPassive"] else "verb"
                    clause_slots["abst-obj"] = abst_noun(clause_slots["obj"])
                    clause_slots["abst-prep1-obj"] = abst_noun(clause_slots["prep1-obj"])
                    clause_slots["abst-prep2-obj"] = abst_noun(clause_slots["prep2-obj"])
                    clause_slots["abst-misc"] = abst_noun(clause_slots["misc"])
                    clause_slots_dict = { ("clause-%s" % k) : get_clause_slot_field(k, v) for k, v in clause_slots.items() }
                except KeyError:
                    logger.info("Omitting instance without clause data: %s / %s / %s" % (sentence_id, verb_index, question_label["questionString"]))
                    continue
            else:
                clause_slots_dict = {}

            all_slots_dict = {**question_slots_dict, **abstract_slots_dict, **clause_slots_dict}
            included_slot_fields = { k: v for k, v in all_slots_dict.items() if k in self._slot_names }

            answer_fields = get_answer_fields(question_label, verb_fields["text"])

            metadata = {
                "question_label": question_label,
                "gold_spans": set(get_answer_spans([question_label])[0])
            }

            yield {
                **verb_fields, **included_slot_fields, **answer_fields,
                "metadata": metadata
            }

@QasrlInstanceReader.register("question_with_sentence_single_span")
class QasrlQuestionReader(QasrlInstanceReader):
    def __init__(self):
        self._tokenizer = WordTokenizer()
    @overrides
    def read_instances(self,
                       token_indexers: Dict[str, TokenIndexer],
                       sentence_id: str,
                       sentence_tokens: List[str],
                       verb_index: int,
                       verb_inflected_forms: Dict[str, str],
                       question_labels): # Iterable[Dict[str, ?Field]]
        for question_label in question_labels:
            question_tokens = self._tokenizer.tokenize(question_label["questionString"])
            question_with_sentence_tokens = question_tokens + [Token("[SEP]")] + [Token(t) for t in sentence_tokens] + [Token("CANNOTANSWER")]
            question_with_sentence_field = TextField(question_with_sentence_tokens, token_indexers)
            span_index_offset = len(question_tokens) + 1
            invalid_index = len(question_with_sentence_tokens) - 1

            num_invalids = get_num_invalids(question_label)
            answer_spans = get_answer_spans([question_label])
            offset_answer_spans = [
                Span(s.start() + span_index_offset, s.end() + span_index_offset)
                for s in answer_spans
            ]
            if num_invalids > 0 or len(answer_spans) == 0:
                start_index_field = IndexField(invalid_index, question_with_sentence_field)
                end_index_field = IndexField(invalid_index, question_with_sentence_field)
            else:
                random_span = offset_answer_spans[random.randint(0, len(answer_spans) - 1)]
                start_index_field = IndexField(random_span.start(), question_with_sentence_field)
                end_index_field = IndexField(random_span.end(), question_with_sentence_field)

            metadata = {
                "question_label": question_label,
                "gold_spans": set(offset_answer_spans),
                "span_index_offset": span_index_offset
            }

            yield {
                "text": question_with_sentence_field,
                "start_index": start_index_field,
                "end_index": end_index_field,
                "metadata": metadata
            }

@QasrlInstanceReader.register("span_animacy")
class QasrlAnimacyReader(QasrlInstanceReader):
    def __init__(self):
        self._tokenizer = WordTokenizer()
    @overrides
    def read_instances(self,
                       token_indexers: Dict[str, TokenIndexer],
                       sentence_id: str,
                       sentence_tokens: List[str],
                       verb_index: int,
                       verb_inflected_forms: Dict[str, str],
                       question_labels): # Iterable[Dict[str, ?Field]]
        verb_fields = get_verb_fields(token_indexers, sentence_tokens, verb_index)
        animacy_label_map = {}
        for question_label in question_labels:
            if question_label["questionSlots"]["wh"] in { "who", "what" }:
                is_animate = question_label["questionSlots"]["wh"] == "who"
                animacy_label = 1 if is_animate else 0
                for span in get_answer_spans([question_label]):
                    if span not in animacy_label_map:
                        animacy_label_map[span] = []
                    animacy_label_map[span].append(animacy_label)

        animacy_span_fields = []
        animacy_label_fields = []
        for s, labels in animacy_label_map.items():
            if len(labels) > 0 and not (0 in labels and 1 in labels):
                animacy_span_fields.append(SpanField(s.start(), s.end(), verb_fields["text"]))
                animacy_label_fields.append(LabelField(label = labels[0], skip_indexing = True))

        if len(animacy_span_fields) > 0:
            yield {
                **verb_fields,
                "animacy_spans": ListField(animacy_span_fields),
                "animacy_labels": ListField(animacy_label_fields)
            }

@QasrlInstanceReader.register("span_tan")
class QasrlSpanTanReader(QasrlInstanceReader):
    def __init__(self):
        self._tokenizer = WordTokenizer()
    @overrides
    def read_instances(self,
                       token_indexers: Dict[str, TokenIndexer],
                       sentence_id: str,
                       sentence_tokens: List[str],
                       verb_index: int,
                       verb_inflected_forms: Dict[str, str],
                       question_labels): # Iterable[Dict[str, ?Field]]
        verb_fields = get_verb_fields(token_indexers, sentence_tokens, verb_index)
        tan_label_map = {}
        for question_label in question_labels:
            tan_string = get_tan_string(question_label)
            for span in get_answer_spans([question_label]):
                if span not in tan_label_map:
                    tan_label_map[span] = []
                tan_label_map[span].append(tan_string)

        tan_span_fields = []
        tan_label_fields = []
        for s, labels in tan_label_map.items():
            if len(labels) > 0:
                tan_span_fields.append(SpanField(s.start(), s.end(), verb_fields["text"]))
                tan_label_fields.append(MultiLabelField_New(list(set(labels)), label_namespace = "tan-string-labels"))

        if len(tan_span_fields) > 0:
            yield {
                **verb_fields,
                "tan_spans": ListField(tan_span_fields),
                "tan_labels": ListField(tan_label_fields)
            }

@QasrlInstanceReader.register("clause_dist")
class QasrlQuestionFactoredReader(QasrlInstanceReader):
    def __init__(self,
                 clause_info_files: List[str] = []):
        self._clause_info = None
        if len(clause_info_files) > 0:
            self._clause_info = {}
            for file_path in clause_info_files:
                read_clause_info(self._clause_info, file_path)
        self._tokenizer = WordTokenizer()
    @overrides
    def read_instances(self,
                       token_indexers: Dict[str, TokenIndexer],
                       sentence_id: str,
                       sentence_tokens: List[str],
                       verb_index: int,
                       verb_inflected_forms: Dict[str, str],
                       question_labels): # Iterable[Dict[str, ?Field]]
        verb_fields = get_verb_fields(token_indexers, sentence_tokens, verb_index)
        clause_dist_fields = {}
        if self._clause_info is not None:
            clause_counter = Counter()
            for question_label in question_labels:
                clause_slots = {}
                try:
                    clause_slots = self._clause_info[sentence_id][verb_index][question_label["questionString"]]["slots"]
                except KeyError:
                    logger.info("Omitting instance without clause data: %s / %s / %s" % (sentence_id, verb_index, question_label["questionString"]))
                    continue

                def abst_noun(x):
                    return "something" if (x == "someone") else x
                clause_slots["abst-subj"] = abst_noun(clause_slots["subj"])
                clause_slots["abst-verb"] = "verb[pss]" if question_label["isPassive"] else "verb"
                clause_slots["abst-obj"] = abst_noun(clause_slots["obj"])
                clause_slots["abst-prep1-obj"] = abst_noun(clause_slots["prep1-obj"])
                clause_slots["abst-prep2-obj"] = abst_noun(clause_slots["prep2-obj"])
                clause_slots["abst-misc"] = abst_noun(clause_slots["misc"])
                abst_slot_names = ["abst-subj", "abst-verb", "abst-obj", "prep1", "abst-prep1-obj", "prep2", "abst-prep2-obj", "abst-misc"]
                clause_string = " ".join([clause_slots[slot_name] for slot_name in abst_slot_names])
                clause_counter[clause_string] += 1

            clause_dist_fields["clause_dist"] = MultisetField(labels = clause_counter, label_namespace = "abst-clause-labels")

        yield {
            **verb_fields,
            **clause_dist_fields
        }

@QasrlInstanceReader.register("question_factored")
class QasrlQuestionFactoredReader(QasrlInstanceReader):
    def __init__(self,
                 clause_info_files: List[str] = []):
        self._clause_info = None
        if len(clause_info_files) > 0:
            self._clause_info = {}
            for file_path in clause_info_files:
                read_clause_info(self._clause_info, file_path)
        self._tokenizer = WordTokenizer()
    @overrides
    def read_instances(self,
                       token_indexers: Dict[str, TokenIndexer],
                       sentence_id: str,
                       sentence_tokens: List[str],
                       verb_index: int,
                       verb_inflected_forms: Dict[str, str],
                       question_labels): # Iterable[Dict[str, ?Field]]
        verb_fields = get_verb_fields(token_indexers, sentence_tokens, verb_index)
        tan_strings = []
        tan_string_fields = []
        all_answer_fields = []
        if self._clause_info is not None:
            clause_string_fields = []
            clause_strings = []
            qarg_fields = []
            gold_tuples = []

        if len(question_labels) == 0:
            tan_string_list_field = ListField([LabelField(label = -1, label_namespace = "tan-string-labels", skip_indexing = True)])
            clause_string_list_field = ListField([LabelField(label = -1, label_namespace = "abst-clause-labels", skip_indexing = True)])
            qarg_list_field = ListField([LabelField(label = -1, label_namespace = "qarg-labels", skip_indexing = True)])
            answer_spans_field = ListField([ListField([SpanField(-1, -1, verb_fields["text"])])])
            num_answers_field = ListField([LabelField(-1, skip_indexing = True)])
            num_invalids_field = ListField([LabelField(-1, skip_indexing = True)])
        else:
            for question_label in question_labels:

                tan_string = get_tan_string(question_label)
                tan_string_field = LabelField(label = tan_string, label_namespace = "tan-string-labels")
                tan_strings.append(tan_string)
                tan_string_fields.append(tan_string_field)

                answer_fields = get_answer_fields(question_label, verb_fields["text"])
                all_answer_fields.append(answer_fields)

                if self._clause_info is not None:
                    clause_slots = {}
                    try:
                        clause_slots = self._clause_info[sentence_id][verb_index][question_label["questionString"]]["slots"]
                    except KeyError:
                        logger.info("Omitting instance without clause data: %s / %s / %s" % (sentence_id, verb_index, question_label["questionString"]))
                        continue

                    def abst_noun(x):
                        return "something" if (x == "someone") else x
                    clause_slots["abst-subj"] = abst_noun(clause_slots["subj"])
                    clause_slots["abst-verb"] = "verb[pss]" if question_label["isPassive"] else "verb"
                    clause_slots["abst-obj"] = abst_noun(clause_slots["obj"])
                    clause_slots["abst-prep1-obj"] = abst_noun(clause_slots["prep1-obj"])
                    clause_slots["abst-prep2-obj"] = abst_noun(clause_slots["prep2-obj"])
                    clause_slots["abst-misc"] = abst_noun(clause_slots["misc"])
                    abst_slot_names = ["abst-subj", "abst-verb", "abst-obj", "prep1", "abst-prep1-obj", "prep2", "abst-prep2-obj", "abst-misc"]
                    clause_string = " ".join([clause_slots[slot_name] for slot_name in abst_slot_names])
                    clause_string_field = LabelField(label = clause_string, label_namespace = "abst-clause-labels")
                    clause_strings.append(clause_string)
                    clause_string_fields.append(clause_string_field)

                    qarg_fields.append(LabelField(label = clause_slots["qarg"], label_namespace = "qarg-labels"))

                    for span_field in answer_fields["answer_spans"]:
                        if span_field.span_start > -1:
                            s = (span_field.span_start, span_field.span_end)
                            gold_tuples.append((clause_string, clause_slots["qarg"], s))

            tan_string_list_field = ListField(tan_string_fields)
            answer_spans_field = ListField([f["answer_spans"] for f in all_answer_fields])
            num_answers_field = ListField([f["num_answers"] for f in all_answer_fields])
            num_invalids_field = ListField([f["num_invalids"] for f in all_answer_fields])

            if self._clause_info is not None:
                clause_string_list_field = ListField(clause_string_fields)
                qarg_list_field = ListField(qarg_fields)

        if self._clause_info is not None:
            all_clause_strings = set(clause_strings)
            all_spans = set([t[2] for t in gold_tuples])
            all_qargs = set([t[1] for t in gold_tuples])
            qarg_pretrain_clause_fields = []
            qarg_pretrain_span_fields = []
            qarg_pretrain_multilabel_fields = []
            for clause_string in all_clause_strings:
                for span in all_spans:
                    valid_qargs = [qarg for qarg in all_qargs if (clause_string, qarg, span) in gold_tuples]
                    qarg_pretrain_clause_fields.append(LabelField(clause_string, label_namespace = "abst-clause-labels"))
                    qarg_pretrain_span_fields.append(SpanField(span[0], span[1], verb_fields["text"]))
                    qarg_pretrain_multilabel_fields.append(MultiLabelField_New(valid_qargs, label_namespace = "qarg-labels"))

            if len(qarg_pretrain_clause_fields) > 0:
                qarg_labeled_clauses_field = ListField(qarg_pretrain_clause_fields)
                qarg_labeled_spans_field = ListField(qarg_pretrain_span_fields)
                qarg_labels_field = ListField(qarg_pretrain_multilabel_fields)
            else:
                qarg_labeled_clauses_field = ListField([LabelField(-1, label_namespace = "abst-clause-labels", skip_indexing = True)])
                qarg_labeled_spans_field = ListField([SpanField(-1, -1, verb_fields["text"])])
                qarg_labels_field = ListField([MultiLabelField_New(set(), label_namespace = "qarg-labels")])

        tan_multilabel_field = MultiLabelField_New(list(set(tan_strings)), label_namespace = "tan-string-labels")

        if self._clause_info is not None:
            yield {
                **verb_fields,
                "clause_strings": clause_string_list_field,
                "clause_set": MultiLabelField_New(clause_strings, label_namespace = "abst-clause-labels"),
                "tan_strings": tan_string_list_field,
                "tan_set": tan_multilabel_field,
                "qargs": qarg_list_field,
                "answer_spans": answer_spans_field,
                "num_answers": num_answers_field,
                "num_invalids": num_invalids_field,
                "metadata": MetadataField({
                    "gold_set": set(gold_tuples) # TODO make it a multiset so we can change span selection policy?
                }),
                "qarg_labeled_clauses": qarg_labeled_clauses_field,
                "qarg_labeled_spans": qarg_labeled_spans_field,
                "qarg_labels": qarg_labels_field,
            }
        else:
            yield {
                **verb_fields,
                "tan_strings": tan_string_list_field,
                "tan_set": tan_multilabel_field,
                "answer_spans": answer_spans_field,
                "num_answers": num_answers_field,
                "num_invalids": num_invalids_field,
                "metadata": MetadataField({}),
            }
