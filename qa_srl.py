import json
import os, stat
import spacy
import nltk
nltk.download('punkt')

from nltk.corpus import stopwords
# nltk.download('stopwords')
import nltk
nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import word_tokenize

nlp = spacy.load('en_core_web_sm')
from shutil import copyfile
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
should_take_only_top1_question = False
verbose = False
should_run_qasrl = True


from coref import pronouns


def main():
    if should_run_qasrl:
        write_qasrl_output_files(coref_text_files_dir)


def prepare_file_to_qasrl(src, dst):
    input = open(src, 'r')
    output = open(dst, 'w')

    for i, line in enumerate(input):
        new_line = str(i + 1) + '\t' + line
        output.write(new_line)
    input.close()
    output.close()


def write_qasrl_output_files(text_file_names):
    for coref_text_file in text_file_names:
        coref_text_file_after_qasrl = os.path.join(coref_text_files_after_qasrl_dir, coref_text_file.replace(".txt", "") +
                                                   "_span_to_question.jsonl")
        src = os.path.join(coref_text_files_dir, coref_text_file)
        prepare_file_to_qasrl(src, qasrl_input_file_path)

        os.chmod(qasrl_input_file_path, stat.S_IRWXU)
        command = "python " + prepare_input_file_path + " --input " + qasrl_input_file_path + " --output " + qasrl_output_file_path
        os.system(command)

        os.chmod(afirst_pipeline_sequential_file_path, stat.S_IRWXU)
        cuda_device, span_min_prob, question_min_prob, question_beam_size = "-1", "0.02", "0.01", "20"
        command = "python " + afirst_pipeline_sequential_file_path + \
                  " --span " + span_density_softmax_file_path + " --span_to_question " + \
                  span_to_question_file_path + " --cuda_device " + cuda_device + " --span_min_prob " + \
                  span_min_prob + " --question_min_prob " + question_min_prob + " --question_beam_size " + \
                  question_beam_size + " --input_file " + qasrl_output_file_path + " --output_file " + coref_text_file_after_qasrl
        os.system(command)


def get_question_from_questions_slots(question_slots, verb):
    result = []
    verb_time = None
    for key, val in question_slots.items():
        if val != '_':
            if key == 'verb':
                val_list = val.split(' ')
                if len(val_list) > 1:
                    verb_to_append = []
                    for word in val_list:
                        if word in verb['verbInflectedForms'] and val_list[0] in ['be', 'being']:
                            verb_time = word
                            verb_to_append.append(verb['verbInflectedForms'][word])
                        else:
                            verb_to_append.append(word)
                    result.append(" ".join(verb_to_append))
                else:
                    result.append(verb['verbInflectedForms'][val])
            else:
                result.append(val)

    return question_slots, " ".join(result).lower() + "?", verb_time


def entity_contains_verb(sentence_tokens, sentence_verbs_indices, entity):
    verbs = [sentence_tokens[idx] for idx in sentence_verbs_indices]
    for word in entity.split(' '):
        if word in verbs:
            return True
    return False

def is_doc_contains_noun(doc):
    for token in doc:
        if token.pos_ in ["PROPN", "NOUN"]:
            return True
    return False

def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent

def entity_contains_noun(entity):
    tuples = preprocess(entity)
    for tup in tuples:
        if tup[1] in ["NNP", "NN", "NNS", "NNPS"]:
            return True
    return False
    # text_tokens = word_tokenize(entity.lower())
    # entity_without_stopwords = " ".join([word for word in text_tokens if not word in stopwords.words()])
    # return is_doc_contains_noun(nlp(entity)) or is_doc_contains_noun(nlp(entity_without_stopwords))




def populate_beams_before_after_verb_lists(verb):
    beams_before_verb, beams_after_verb = [], []
    for beam in verb['beam']:
        span_start = beam['span'][0]
        span_end = beam['span'][1]
        if span_end <= verb['verbIndex']:
            beams_before_verb.append(beam)
        elif span_start > verb['verbIndex']:
            beams_after_verb.append(beam)
    return beams_before_verb, beams_after_verb


def create_answer_question_map(question_answers_map):
    answer_question_map = {}
    for key, val in question_answers_map.items():
        for item in val:
            if item in answer_question_map:
                answer_question_map[item].append(key)
            else:
                answer_question_map[item] = [key]
    return answer_question_map


def shouldIgnoreQA(question, q_slots, verb, ans_prob, entity, sentence_tokens, sentence_verbs_indices):
    if question['questionProb'] <= q_prob_threshold:
        if verbose:
            print("Filter out QA because question probability is: " + str(question['questionProb']) + " which is less or equal to: " + str(q_prob_threshold))
        return True
    if question['questionSlots']['wh'] not in possible_questions:
        if verbose:
            print("Filter out QA because question type is: " + question['questionSlots']['wh'] + " which is not supported")
        return True

    if ans_prob <= ans_prob_threshold:
        if verbose:
            print("Filter out QA because of answer probability is: " + str(ans_prob) + " which is less or equal to: " + str(ans_prob_threshold))
        return True

    # q_slots['wh'] != '_' and q_slots['aux'] == '_' and q_slots['subj'] == "_" \
    # and q_slots['obj'] == "_" and q_slots['prep'] == "_" and q_slots['obj2'] == "_" \
    # and
    if verb['verbInflectedForms']['stem'].lower() == "be":
        if verbose:
            print("Filter out QA because this question contains only 'be' verb")
        return True

    if q_slots['subj'] != '_' and q_slots['verb'] != '_' and q_slots['obj'] != '_':
        if verbose:
            print("Filter out QA because this question contains subj+verb+obj")
        return True

    if entity_contains_verb(sentence_tokens, sentence_verbs_indices, entity):
        if verbose:
            print("Filter out QA because entity contains verb: " + entity)
        return True

    if not entity_contains_noun(entity):

        if verbose:
            print("Filter out QA because entity does not contains noun: " + entity)
        return True

    if entity.lower() in pronouns:
        if verbose:
            print("Filter out QA because entity is a pronoun: " + entity)
        return True

    return False


def update_question_answers_map(question_answers_map, q, q_sub_verb_obj, original_verb, side, line_idx, verb_idx, entity):
    if (q, q_sub_verb_obj, original_verb, side, line_idx + 1, verb_idx + 1) in question_answers_map:
        question_answers_map[(q, q_sub_verb_obj, original_verb, side, line_idx + 1, verb_idx + 1)].append(
            entity.lower())
    else:
        question_answers_map[(q, q_sub_verb_obj, original_verb, side, line_idx + 1, verb_idx + 1)] = [
            entity.lower()]


def process_beams(beams, before_after, sentence_tokens, verb, sentence_verbs_indices):
    q_list, q_sub_verb_obj_list, entity_list = [], [], []
    for beam in beams:
        for question in beam['questions']:
            q_slots, q, verb_time = get_question_from_questions_slots(question['questionSlots'], verb)
            q_verb = '<verb>' if q_slots['verb'] != '_' else '_'
            q_subj = '<subj>' if q_slots['subj'] != '_' else '_'
            q_obj = '<obj>' if q_slots['obj'] != '_' else '_'
            q_sub_verb_obj = q_subj + q_verb + q_obj
            entity = " ".join(sentence_tokens[beam['span'][0]:beam['span'][1]])
            if verbose:
                print("Entity: " + before_after + " " + entity)

            ans_prob = round(beam['spanProb'], 2)
            q_prob = round(question['questionProb'], 2)

            if verbose:
                print("question with answer " + before_after + " verb: " + q + "\nanswer: " + entity + ", answer prob:" + str(
                    ans_prob) + ", question_prob:" + str(q_prob))

            if shouldIgnoreQA(question, q_slots, verb, ans_prob, entity, sentence_tokens,
                              sentence_verbs_indices):
                continue

            q_list.append(q)
            q_sub_verb_obj_list.append(q_sub_verb_obj)
            entity_list.append(entity)

            if should_take_only_top1_question:
                continue


    return q_list, q_sub_verb_obj_list, entity_list


def read_parsed_qasrl(filename):
    f = open(filename, "r")
    lines = f.readlines()
    question_answers_map = {}
    for line_idx, line in enumerate(lines):
        json_object = json.loads(line)
        sentence_id = json_object['sentenceId']
        sentence_tokens = json_object['sentenceTokens']
        if verbose:
            print("sentence " + sentence_id + ": " + " ".join(sentence_tokens))

        sentence_verbs_indices = [d['verbIndex'] for d in json_object['verbs']]
        for verb_idx, verb in enumerate(json_object['verbs']):
            original_verb, stemmed_verb = sentence_tokens[verb['verbIndex']].lower(), verb['verbInflectedForms']['stem'].lower()
            if verbose:
                print("verb " + str(verb_idx+1) + " (original): " + original_verb)
                print("verb " + str(verb_idx+1) + " (stem): " + stemmed_verb)

            beams_before_verb, beams_after_verb = populate_beams_before_after_verb_lists(verb)

            q_list, q_sub_verb_obj_list, entity_before_list = process_beams(beams_before_verb, "before", sentence_tokens, verb, sentence_verbs_indices)
            for i in range(len(q_list)):
                update_question_answers_map(question_answers_map, q_list[i], q_sub_verb_obj_list[i], original_verb, 'L', line_idx, verb_idx, entity_before_list[i].strip())

            q_list, q_sub_verb_obj_list, entity_after_list = process_beams(beams_after_verb, "after", sentence_tokens, verb, sentence_verbs_indices)
            for i in range(len(q_list)):
                update_question_answers_map(question_answers_map, q_list[i], q_sub_verb_obj_list[i], original_verb, 'R', line_idx, verb_idx, entity_after_list[i].strip())

    answer_question_map = create_answer_question_map(question_answers_map)
    return answer_question_map


if __name__ == '__main__':
    main()
