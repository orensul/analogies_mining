import json
import os, stat
import spacy
import nltk
from coref import pronouns

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nlp = spacy.load('en_core_web_sm')


qasrl_input_file_path = 'data/input.txt'
qasrl_output_file_path = 'data/out.jsonl'

coref_text_files_dir = '../data/coref_text_files'
coref_text_files_after_qasrl_dir = './data'

prepare_input_file_path = "./scripts/utils/prepare_input_file.py"
afirst_pipeline_sequential_file_path = "./qasrl/pipelines/afirst_pipeline_sequential.py"
span_density_softmax_file_path = "./models/span_density_softmax.tar.gz"
span_to_question_file_path = "./models/span_to_question.tar.gz"


possible_questions = {'what', 'who', 'which'}

ans_prob_threshold = 0.05
q_prob_threshold = 0.1
span_min_prob = 0.02
question_min_prob = 0.01
question_beam_size = 20
cuda_device = -1

should_take_only_top1_question = False
verbose = False


def main():
    write_qasrl_output_files(coref_text_files_dir)


# Running QA-SRL offline -- one time


def write_qasrl_output_files(text_file_names):
    """
    Run QA-SRL on the text files from data/coref_text_files and outputs json files in qasrl-modeling/data/
    """
    for coref_text_file in text_file_names:
        coref_text_file_after_qasrl = os.path.join(coref_text_files_after_qasrl_dir, coref_text_file.replace(".txt", "") +
                                                   "_span_to_question.jsonl")
        src = os.path.join(coref_text_files_dir, coref_text_file)
        prepare_file_to_qasrl(src, qasrl_input_file_path)

        os.chmod(qasrl_input_file_path, stat.S_IRWXU)
        command = "python " + prepare_input_file_path + " --input " + qasrl_input_file_path + " --output " + qasrl_output_file_path
        os.system(command)

        os.chmod(afirst_pipeline_sequential_file_path, stat.S_IRWXU)
        command = "python " + afirst_pipeline_sequential_file_path + \
                  " --span " + span_density_softmax_file_path + " --span_to_question " + \
                  span_to_question_file_path + " --cuda_device " + str(cuda_device) + " --span_min_prob " + \
                  str(span_min_prob) + " --question_min_prob " + str(question_min_prob) + " --question_beam_size " + \
                  str(question_beam_size) + " --input_file " + qasrl_output_file_path + " --output_file " + coref_text_file_after_qasrl
        os.system(command)


def prepare_file_to_qasrl(src, dst):
    """
    Prepare the input for QA-SRL (adding line number, tab and the sentence for every sentence in the text)
    """
    input = open(src, 'r')
    output = open(dst, 'w')

    for i, line in enumerate(input):
        new_line = str(i + 1) + '\t' + line
        output.write(new_line)
    input.close()
    output.close()


# Use QA-SRL online -- every time we run find_mappings.py or find_mappings_verbs.py

# Parse QA-SRL for FMQ method

def read_parsed_qasrl(filename):
    """
    read and parse the output QA-SRL json for FMQ method and returns answer_questions_map
    """
    f = open(filename, "r")
    lines = f.readlines()
    question_answers_map = {}
    for line_idx, line in enumerate(lines):
        json_object = json.loads(line)
        sentence_id, sentence_tokens = json_object['sentenceId'], json_object['sentenceTokens']

        if verbose:
            print("sentence " + sentence_id + ": " + " ".join(sentence_tokens))

        sentence_verbs_indices = [d['verbIndex'] for d in json_object['verbs']]
        for verb_idx, verb in enumerate(json_object['verbs']):
            original_verb, stemmed_verb = sentence_tokens[verb['verbIndex']].lower(), verb['verbInflectedForms']['stem'].lower()
            if verbose:
                print("verb " + str(verb_idx+1) + " (original): " + original_verb + "\n" + "verb " + str(verb_idx+1) + " (stem): " + stemmed_verb)

            beams_before_verb, beams_after_verb = populate_beams_before_after_verb_lists(verb)
            q_list, q_sub_verb_obj_list, entity_before_list = process_beams(beams_before_verb, "before", sentence_tokens, verb, sentence_verbs_indices)
            for i in range(len(q_list)):
                update_question_answers_map(question_answers_map, q_list[i], q_sub_verb_obj_list[i], original_verb, 'L', line_idx, verb_idx, entity_before_list[i].strip())

            q_list, q_sub_verb_obj_list, entity_after_list = process_beams(beams_after_verb, "after", sentence_tokens, verb, sentence_verbs_indices)
            for i in range(len(q_list)):
                update_question_answers_map(question_answers_map, q_list[i], q_sub_verb_obj_list[i], original_verb, 'R', line_idx, verb_idx, entity_after_list[i].strip())

    answer_questions_map = create_answer_questions_map(question_answers_map)
    return answer_questions_map


def populate_beams_before_after_verb_lists(verb):
    """
    Returns the beams to the left of the verb (before) and to the right (after)
    """
    beams_before_verb, beams_after_verb = [], []
    for beam in verb['beam']:
        span_start = beam['span'][0]
        span_end = beam['span'][1]
        if span_end <= verb['verbIndex']:
            beams_before_verb.append(beam)
        elif span_start > verb['verbIndex']:
            beams_after_verb.append(beam)
    return beams_before_verb, beams_after_verb


def process_beams(beams, before_after, sentence_tokens, verb, sentence_verbs_indices):
    """
    Returns list of questions(q_list), list of the structure of the questions(q_sub_obj_list) and list of entities(entity_list)
    by parsing the QA-SRL json, reading the questions and answers, and applying some filter rules to ignore some QA.
    """
    q_list, q_sub_verb_obj_list, entity_list = [], [], []
    for beam in beams:
        for question in beam['questions']:
            q_slots, q = get_question_from_questions_slots(question['questionSlots'], verb)
            q_verb = '<verb>' if q_slots['verb'] != '_' else '_'
            q_subj = '<subj>' if q_slots['subj'] != '_' else '_'
            q_obj = '<obj>' if q_slots['obj'] != '_' else '_'
            q_sub_verb_obj = q_subj + q_verb + q_obj
            entity = " ".join(sentence_tokens[beam['span'][0]:beam['span'][1]])

            if verbose:
                print("Entity: " + before_after + " " + entity)

            ans_prob, q_prob = round(beam['spanProb'], 2), round(question['questionProb'], 2)

            if verbose:
                print("question with answer " + before_after + " verb: " + q + "\nanswer: " + entity + ", answer prob:" + str(
                    ans_prob) + ", question_prob:" + str(q_prob))

            if should_ignore_qa(question, q_slots, verb, ans_prob, entity, sentence_tokens, sentence_verbs_indices):
                continue

            q_list.append(q)
            q_sub_verb_obj_list.append(q_sub_verb_obj)
            entity_list.append(entity)

            if should_take_only_top1_question:
                continue

    return q_list, q_sub_verb_obj_list, entity_list


def get_question_from_questions_slots(question_slots, verb):
    """
    Generate the question by concatenating the parts of the QA-SRL 7-slot template.
    Returns the question_slots and the generated question.
    """
    result = []
    for key, val in question_slots.items():
        if val != '_':
            if key == 'verb':
                val_list = val.split(' ')
                if len(val_list) > 1:
                    verb_to_append = []
                    for word in val_list:
                        if word in verb['verbInflectedForms'] and val_list[0] in ['be', 'being']:
                            verb_to_append.append(verb['verbInflectedForms'][word])
                        else:
                            verb_to_append.append(word)
                    result.append(" ".join(verb_to_append))
                else:
                    result.append(verb['verbInflectedForms'][val])
            else:
                result.append(val)

    return question_slots, " ".join(result).lower() + "?"


def should_ignore_qa(question, q_slots, verb, ans_prob, entity, sentence_tokens, sentence_verbs_indices):
    """
    Returns boolean whether to ignore the QA or not (Appendix A in the paper)
    """
    if question['questionProb'] <= q_prob_threshold:
        if verbose:
            print("Filter QA because question probability is: " + str(question['questionProb']) + " which is less or equal to: " + str(q_prob_threshold))
        return True

    if question['questionSlots']['wh'] not in possible_questions:
        if verbose:
            print("Filter QA because question type is: " + question['questionSlots']['wh'] + " which is not supported")
        return True

    if ans_prob <= ans_prob_threshold:
        if verbose:
            print("Filter QA because answer probability is: " + str(ans_prob) + " which is less or equal to: " + str(ans_prob_threshold))
        return True

    if verb['verbInflectedForms']['stem'].lower() == "be":
        if verbose:
            print("Filter QA because this question contains 'be' verb")
        return True

    if q_slots['subj'] != '_' and q_slots['verb'] != '_' and q_slots['obj'] != '_':
        if verbose:
            print("Filter QA because this question contains subj+verb+obj")
        return True

    if entity_contains_verb(sentence_tokens, sentence_verbs_indices, entity):
        if verbose:
            print("Filter QA because entity contains verb: " + entity)
        return True

    if not entity_contains_noun(entity):
        if verbose:
            print("Filter QA because entity does not contains noun: " + entity)
        return True

    if entity.lower() in pronouns:
        if verbose:
            print("Filter QA because entity is a pronoun: " + entity)
        return True

    return False


def entity_contains_verb(sentence_tokens, sentence_verbs_indices, entity):
    verbs = [sentence_tokens[idx] for idx in sentence_verbs_indices]
    for word in entity.split(' '):
        if word in verbs:
            return True
    return False


def entity_contains_noun(entity):
    tuples = preprocess(entity)
    for tup in tuples:
        if tup[1] in ["NNP", "NN", "NNS", "NNPS"]:
            return True
    return False


def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent


def create_answer_questions_map(question_answers_map):
    """
    Returns a new dictionary with answer(entity) as key and questions as value, from the opposite dictionary
    (question as key and answers as value)
    """
    answer_questions_map = {}
    for question, answers in question_answers_map.items():
        for ans in answers:
            if ans in answer_questions_map:
                answer_questions_map[ans].append(question)
            else:
                answer_questions_map[ans] = [question]
    return answer_questions_map


def update_question_answers_map(question_answers_map, q, q_sub_verb_obj, original_verb, side, line_idx, verb_idx, entity):
    """
    Update question_answers_map with the answer(entity) corresponding to the question.
    """
    if (q, q_sub_verb_obj, original_verb, side, line_idx + 1, verb_idx + 1) in question_answers_map:
        question_answers_map[(q, q_sub_verb_obj, original_verb, side, line_idx + 1, verb_idx + 1)].append(entity.lower())
    else:
        question_answers_map[(q, q_sub_verb_obj, original_verb, side, line_idx + 1, verb_idx + 1)] = [entity.lower()]


# Parse QA-SRL for FMV method


def read_parsed_qasrl_verbs(filename):
    """
    read and parse the output QA-SRL json for FMV method and returns answer_verbs_map
    """
    f = open(filename, "r")
    lines = f.readlines()
    verb_answers_map = {}
    for line_idx, line in enumerate(lines):
        json_object = json.loads(line)
        sentence_id, sentence_tokens = json_object['sentenceId'], json_object['sentenceTokens']

        if verbose:
            print("sentence " + sentence_id + ": " + " ".join(sentence_tokens))

        sentence_verbs_indices = [d['verbIndex'] for d in json_object['verbs']]
        for verb_idx, verb in enumerate(json_object['verbs']):
            original_verb, stemmed_verb = sentence_tokens[verb['verbIndex']].lower(), verb['verbInflectedForms'][
                'stem'].lower()
            if verbose:
                print("verb " + str(verb_idx + 1) + " (original): " + original_verb + "\n" + "verb " + str(
                    verb_idx + 1) + " (stem): " + stemmed_verb)

            beams_before_verb, beams_after_verb = populate_beams_before_after_verb_lists(verb)
            verbs, entities = process_beams_verbs(beams_before_verb, sentence_tokens, verb, sentence_verbs_indices)
            update_verb_answers_map(verb_answers_map, verbs, entities)

            verbs, entities = process_beams_verbs(beams_after_verb, sentence_tokens, verb, sentence_verbs_indices)
            update_verb_answers_map(verb_answers_map, verbs, entities)

    answer_verbs_map = create_answer_verbs_map(verb_answers_map)
    return answer_verbs_map


def process_beams_verbs(beams, sentence_tokens, verb, sentence_verbs_indices):
    """
    Returns a list of verbs, and a list of answers(entities)
    by parsing the QA-SRL json, reading the questions and answers, extracting the verbs from questions
    and applying some filter rules to ignore some QA.
    """
    verbs, entities = [], []
    for beam in beams:
        ent = " ".join(sentence_tokens[beam['span'][0]:beam['span'][1]])
        ans_prob = round(beam['spanProb'], 2)
        if should_ignore_qa_verbs(ans_prob, ent, sentence_tokens, sentence_verbs_indices):
            continue
        entities.append(ent)
        verbs.append(sentence_tokens[verb['verbIndex']].lower())

    return verbs, entities


def should_ignore_qa_verbs(ans_prob, entity, sentence_tokens, sentence_verbs_indices):
    """
    Returns boolean if to filter QA or not for the FMV method (here the questions are not relevant)
    """
    if ans_prob <= ans_prob_threshold:
        if verbose:
            print("Filter QA because of answer probability is: " + str(ans_prob) + " which is less or equal to: " + str(ans_prob_threshold))
        return True

    if entity_contains_verb(sentence_tokens, sentence_verbs_indices, entity):
        if verbose:
            print("Filter QA because entity contains verb: " + entity)
        return True

    if not entity_contains_noun(entity):
        if verbose:
            print("Filter QA because entity does not contains noun: " + entity)
        return True

    if entity.lower() in pronouns:
        if verbose:
            print("Filter QA because entity is a pronoun: " + entity)
        return True

    return False


def update_verb_answers_map(verb_answers_map, verbs, entities):
    """
    Update the dictionary of verb_answers_map with the answers(entities) corresponding to the verbs
    """
    for i in range(len(verbs)):
        verb, entity = verbs[i], entities[i]
        if verb not in verb_answers_map:
            verb_answers_map[verb] = [entity]
        else:
            verb_answers_map[verb].append(entity)


def create_answer_verbs_map(verb_answers_map):
    """
    Returns a new dictionary with answer(entity) as key and verbs as value, from the opposite dictionary
    (verb as key and answers as value)
    """
    answer_verbs_map = {}
    for verb, answers in verb_answers_map.items():
        for ans in answers:
            if ans in answer_verbs_map:
                answer_verbs_map[ans].append(verb)
            else:
                answer_verbs_map[ans] = [verb]
    return answer_verbs_map


if __name__ == '__main__':
    main()
