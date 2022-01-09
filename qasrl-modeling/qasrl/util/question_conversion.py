from typing import Dict

from allennlp.data.vocabulary import Vocabulary

import torch

from qasrl.data.util import get_slot_label_namespace

def get_abst_question_slots(slots):
    def replace_for_slot(slot, old, new):
        return new if slots[slot] == old else slots[slot]
    is_passive = "pastParticiple" in slots["verb"] and (
        "be" in slots["verb"] or
        "been" in slots["verb"] or
        "being" in slots["verb"] or
        "is" in slots["aux"] or
        "was" in slots["aux"])
    return {
        "abst-wh": replace_for_slot("wh", "who", "what"),
        "abst-subj": replace_for_slot("subj", "someone", "something"),
        "abst-verb": "verb[pss]" if is_passive else "verb",
        "abst-obj": replace_for_slot("obj", "someone", "something"),
        "abst-prep": "_" if slots["prep"] == "_" else "<prep>",
        "abst-obj2": replace_for_slot("obj2", "someone", "something")
    }

_placeholder_and_gap_mapping = {
    "someone": ("who", "_"),
    "something": ("what", "_"),
    "doing something": ("what", "doing"),
    "to do something": ("what", "to do"),
    "do something": ("what", "do"),
    "somewhere": ("where", "_"),
}

def get_wh_for_slot_value(slot_value):
    if slot_value not in _placeholder_and_gap_mapping:
        raise ValueError("Answer can only be a non-empty slot")
    return _placeholder_and_gap_mapping[slot_value][0]
def get_gap_for_slot_value(slot_value):
    if slot_value not in _placeholder_and_gap_mapping:
        raise ValueError("Gap can only be taken for a non-empty slot")
    return _placeholder_and_gap_mapping[slot_value][1]

def get_question_for_clause(clause_slots, vocab: Vocabulary):
    answer_slot = clause_slots["qarg"]
    wh = answer_slot if answer_slot not in clause_slots else get_wh_for_slot_value(clause_slots[answer_slot])
    subj = clause_slots["subj"] if answer_slot != "subj" else get_gap_for_slot_value(clause_slots["subj"])
    if clause_slots["aux"] == "_" and subj != "_":
        verb = "stem"
        if clause_slots["verb"] == "past":
            aux = "did"
        elif clause_slots["verb"] == "present":
            aux = "does"
        else:
            raise ValueError("Verb slot %s cannot be split" % clause_slots["verb"])
    else:
        aux = clause_slots["aux"]
        verb = clause_slots["verb"] if clause_slots["verb"] != "present" else "presentSingular3rd"

    obj = clause_slots["obj"] if answer_slot != "obj" else get_gap_for_slot_value(clause_slots["obj"])

    if clause_slots["prep1"] != "_" and clause_slots["prep2"] != "_":
        prep = "%s %s" % (clause_slots["prep1"], clause_slots["prep2"])
        try:
            vocab.get_token_index(prep, namespace = get_slot_label_namespace("prep"))
        except KeyError:
            raise ValueError("Preposition bigram is not in vocabulary: %s" % prep)
        if clause_slots["prep1-obj"] != "_":
            # in something/someone for ...
            if answer_slot != "prep1-obj":
                raise ValueError("First preposition cannot have a placeholder object in the presence of a second preposition")
            prep1_gap = get_gap_for_slot_value(clause_slots["prep1-obj"])
            if prep1_gap != "_":
                raise ValueError("Gapped argument of first preposition in a pair must be empty; was: %s" % prep1_gap)
            # in <gap> for ...
            if clause_slots["prep2-obj"] != "_":
                # in <gap> for someone / (doing) something ...
                if clause_slots["misc"] != "_":
                    raise ValueError("When prep2 object fills last slot, misc must be empty; was: %s" % clause_slots["misc"])
                else:
                    # in <gap> for someone / (doing) something?
                    obj2 = clause_slots["prep2-obj"]
            else:
                # in <gap> for <misc>
                obj2 = clause_slots["misc"]
        else:
            # in for ...
            if clause_slots["prep2-obj"] == "_":
                # in for <misc>
                obj2 = clause_slots["misc"] if answer_slot != "misc" else get_gap_for_slot_value(clause_slots["misc"])
            else:
                # in for ?(someone / (doing) something) ...
                if answer_slot == "prep2-obj":
                    prep2_gap = get_gap_for_slot_value(clause_slots["prep2-obj"])
                    if prep2_gap != "_":
                        if clause_slots["misc"] != "_":
                            raise ValueError("When prep2 gap fills last slot, misc must be empty; was: %s" % clause_slots["misc"])
                        obj2 = prep2_gap
                    else:
                        # in for <gap> ...
                        obj2 = "_" if clause_slots["misc"] == "_" else get_gap_for_slot_value(clause_slots["misc"])
                else:
                    # in for someone / (doing) something ...
                    obj2 = get_gap_for_slot_value(clause_slots["prep2-obj"])
                    if clause_slots["misc"] != "_":
                        if answer_slot != "misc" or get_gap_for_slot_value(clause_slots["misc"]) != "_":
                            raise ValueError("When prep2 object fills last slot, misc must be empty (possibly via a gap); was: %s" % clause_slots["misc"])
    else:
        if clause_slots["prep2"] != "_":
            raise ValueError("Prep2 must only be present when prep1 is; had: %s, %s" % (clause_slots["prep1"], clause_slots["prep2"]))
        # prep1 only: in ...
        prep = clause_slots["prep1"]
        if clause_slots["prep1-obj"] == "_":
            obj2 = "_" if clause_slots["misc"] == "_" else get_gap_for_slot_value(clause_slots["misc"])
        else:
            # in ?(someone / (doing) something)
            if answer_slot == "prep1-obj":
                prep_gap = get_gap_for_slot_value(clause_slots["prep1-obj"])
                if prep_gap != "_":
                    obj2 = prep_gap
                    if clause_slots["misc"] != "_":
                        raise ValueError("When prep2 gap fills last slot, misc must be empty; was: %s" % clause_slots["misc"])
                else:
                    obj2 = "_" if clause_slots["misc"] == "_" else get_gap_for_slot_value(clause_slots["misc"])
            else:
                # in someone / (doing) something
                obj2 = clause_slots["prep1-obj"]
                if clause_slots["misc"] != "_":
                    if answer_slot != "misc" or get_gap_for_slot_value(clause_slots["misc"]) != "_":
                        raise ValueError("When prep2 object fills last slot, misc must be empty (possibly via a gap); was: %s" % clause_slots["misc"])

    res = {
        "wh": wh,
        "aux": aux,
        "subj": subj,
        "verb": verb,
        "obj": obj,
        "prep": prep,
        "obj2": obj2
    }
    # clause_slot_names = ["subj", "aux", "verb", "obj", "prep1", "prep1-obj", "prep2", "prep2-obj", "misc", "qarg"]
    # question_slot_names = ["wh", "aux", "subj", "verb", "obj", "prep", "obj2"]
    # print("===")
    # print("Clause slots:")
    # print(" ".join([clause_slots[slot_name] for slot_name in clause_slot_names]))
    # print("Question slots:")
    # print(" ".join([res[slot_name] for slot_name in question_slot_names]))
    return res

def get_question_tensors_for_clause_tensors_batched(
        batch_size: int,
        vocab: Vocabulary,
        all_slots: Dict[str, torch.LongTensor],
        all_probs: torch.LongTensor):
    clause_slots = { k[len("clause-"):] : v for k, v in all_slots.items() if k.startswith("clause-")}
    question_slot_names = ["wh", "aux", "subj", "verb", "obj", "prep", "obj2"]
    clause_slot_names = ["subj", "aux", "verb", "obj", "prep1", "prep1-obj", "prep2", "prep2-obj", "misc", "qarg"]
    stringy_clause_slots = [
        {k : vocab.get_token_from_index(
                v[i].item(),
                namespace = get_slot_label_namespace("clause-%s" % k))
            for k, v in clause_slots.items()}
        for i in range(batch_size)
    ]
    filtered_stringy_clause_slots = []
    stringy_question_slots = []
    question_probs = []
    # for clause_slots, prob in zip(stringy_clause_slots, all_probs):
    for i in range(len(stringy_clause_slots)):
        try:
            stringy_question_slots.append(get_question_for_clause(stringy_clause_slots[i], vocab))
            filtered_stringy_clause_slots.append(stringy_clause_slots[i])
            question_probs.append(all_probs[i].item())
        except ValueError as e:
            print(str(e))

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:%s" % torch.cuda.current_device())
    filtered_clause_slots = {
        ("clause-%s" % slot_name) : torch.tensor(
            [vocab.get_token_index(slots[slot_name], namespace = get_slot_label_namespace("clause-%s" % slot_name))
             for slots in stringy_clause_slots],
            device = device
        ).long()
        for slot_name in clause_slot_names
    }
    question_slots = {
        slot_name : torch.tensor(
            [vocab.get_token_index(slots[slot_name], namespace = get_slot_label_namespace(slot_name))
             for slots in stringy_question_slots],
            device = device
        ).long()
        for slot_name in question_slot_names
    }
    question_probs_tensor = torch.tensor(question_probs, device = device)
    return filtered_clause_slots, question_slots, question_probs_tensor

