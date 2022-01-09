import json
import os, stat

qasrl_input_file_path = './qasrl-modeling/data/input.txt'
qasrl_output_file_path = './qasrl-modeling/data/out.jsonl'


text_files = ['qasrl-modeling/data/animal_cell.txt', 'qasrl-modeling/data/factory.txt']
text_files_after_qasrl = ['qasrl-modeling/data/animal_cell_span_to_question.jsonl',
                          'qasrl-modeling/data/factory_span_to_question.jsonl']


prepare_input_file_path = "./qasrl-modeling/scripts/utils/prepare_input_file.py"
afirst_pipeline_sequential_file_path = "./qasrl-modeling/qasrl/pipelines/afirst_pipeline_sequential.py"
span_density_softmax_file_path = "./qasrl-modeling/models/span_density_softmax.tar.gz"
span_to_question_file_path = "./qasrl-modeling/models/span_to_question.tar.gz"


possible_questions = {'what', 'who', 'which'}
ans_prob_threshold = 0.05

verbose = False
should_run_qasrl = True


def main():
    if should_run_qasrl:
        write_qasrl_output_files(text_files)


def write_input_qasrl_file(input_file, output_file):
    input_file = open(input_file, 'r')
    output_file = open(output_file, 'w')
    for i, line in enumerate(input_file):
        new_line = str(i+1) + '\t' + line
        output_file.write(new_line)
    input_file.close()
    output_file.close()


def write_qasrl_output_files(text_files):
    for i in range(len(text_files)):
        text_file_name, text_file_after_qasrl = text_files[i], text_files_after_qasrl[i]
        write_input_qasrl_file(text_file_name, qasrl_input_file_path)
        os.chmod(prepare_input_file_path, stat.S_IRWXU)
        command = "python " + prepare_input_file_path + " --input " + qasrl_input_file_path + " --output " + qasrl_output_file_path
        os.system(command)

        os.chmod(afirst_pipeline_sequential_file_path, stat.S_IRWXU)
        cuda_device, span_min_prob, question_min_prob, question_beam_size = "-1", "0.02", "0.01", "20"
        command = "python " + afirst_pipeline_sequential_file_path + \
                  " --span " + span_density_softmax_file_path + " --span_to_question " + \
                  span_to_question_file_path + " --cuda_device " + cuda_device + " --span_min_prob " + \
                  span_min_prob + " --question_min_prob " + question_min_prob + " --question_beam_size " + \
                  question_beam_size + " --input_file " + qasrl_output_file_path + " --output_file " + text_file_after_qasrl
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
        for idx, verb in enumerate(json_object['verbs']):
            verb_idx = verb['verbIndex']
            original_verb, stemmed_verb = sentence_tokens[verb_idx].lower(), verb['verbInflectedForms']['stem'].lower()
            if verbose:
                print("verb " + str(idx+1) + " (original): " + original_verb)
                print("verb " + str(idx+1) + " (stem): " + stemmed_verb)
            beams_before_verb = []
            beams_after_verb = []
            for beam in verb['beam']:
                span_start = beam['span'][0]
                span_end = beam['span'][1]
                if span_end <= verb_idx:
                    beams_before_verb.append(beam)
                elif span_start > verb_idx:
                    beams_after_verb.append(beam)

            for beam_before in beams_before_verb:
                question = beam_before['questions'][0]
                if question['questionSlots']['wh'] not in possible_questions:
                    continue
                q_slots, q, verb_time = get_question_from_questions_slots(question['questionSlots'], verb)
                q_verb = '<verb>' if q_slots['verb'] != '_' else '_'
                q_subj = '<subj>' if q_slots['subj'] != '_' else '_'
                q_obj = '<obj>' if q_slots['obj'] != '_' else '_'
                q_sub_verb_obj = q_subj + q_verb + q_obj
                entity_before = " ".join(sentence_tokens[beam_before['span'][0]:beam_before['span'][1]])
                if verbose:
                    print("Entity before: " + entity_before)

                ans_prob = round(beam_before['spanProb'], 2)
                q_prob = round(question['questionProb'], 2)
                if verbose:
                    print("question with answer before verb: " + q + "\nanswer: " + entity_before + ", answer prob:" + str(
                    ans_prob) + ", question_prob:" + str(q_prob))

                if ans_prob <= ans_prob_threshold:
                    if verbose:
                        print("Filter out QA because of answer probability which is below the threshold: " + entity_before + ", " + q)
                    continue

                if q_slots['subj'] != '_' and q_slots['verb'] != '_' and q_slots['obj'] != '_':
                    if verbose:
                        print("Filter out QA because this question contains subj+verb+obj: " + entity_before + ", " + q)
                    continue
                if entity_contains_verb(sentence_tokens, sentence_verbs_indices, entity_before):
                    if verbose:
                        print("Filter out QA because entity contains verb: " + entity_before + ", " + q)
                    continue

                if (q, q_sub_verb_obj, original_verb, 'L', idx+1) in question_answers_map:
                    question_answers_map[(q, q_sub_verb_obj, original_verb, 'L', idx+1)].append(entity_before.lower())
                else:
                    question_answers_map[(q, q_sub_verb_obj, original_verb, 'L', idx+1)] = [entity_before.lower()]

            for beam_after in beams_after_verb:
                question = beam_after['questions'][0]
                if question['questionSlots']['wh'] not in possible_questions:
                    continue
                q_slots, q, changed_from_passive_to_active = get_question_from_questions_slots(question['questionSlots'], verb)
                q_verb = '<verb>' if q_slots['verb'] != '_' else '_'
                q_subj = '<subj>' if q_slots['subj'] != '_' else '_'
                q_obj = '<obj>' if q_slots['obj'] != '_' else '_'
                q_sub_verb_obj = q_subj + q_verb + q_obj
                entity_after = " ".join(sentence_tokens[beam_after['span'][0]:beam_after['span'][1]])
                if verbose:
                    print("Entity after: " + entity_after)

                ans_prob = round(beam_after['spanProb'], 2)
                q_prob = round(question['questionProb'], 2)
                if verbose:
                    print("question with answer after verb: " + q + "\nanswer: " + entity_after + ", answer prob:" + str(
                    ans_prob) + ", question_prob:" + str(q_prob))

                if ans_prob <= ans_prob_threshold:
                    if verbose:
                        print("Filter out QA because of answer probability which is below the threshold: " + entity_after + ", " + q)
                    continue
                if q_slots['subj'] != '_' and q_slots['verb'] != '_' and q_slots['obj'] != '_':
                    if verbose:
                        print("Filter out QA because this question contains subj+verb+obj: " + entity_after + ", " + q)
                    continue
                if entity_contains_verb(sentence_tokens, sentence_verbs_indices, entity_after):
                    if verbose:
                        print("Filter out QA because entity contains verb: " + entity_after + ", " + q)
                    continue

                if (q, q_sub_verb_obj, original_verb, 'R', idx+1) in question_answers_map:
                    question_answers_map[(q, q_sub_verb_obj, original_verb, 'R', idx+1)].append(entity_after.lower())
                else:
                    question_answers_map[(q, q_sub_verb_obj, original_verb, 'R', idx+1)] = [entity_after.lower()]

    answer_question_map = {}
    for key, val in question_answers_map.items():
        for item in val:
            if item in answer_question_map:
                answer_question_map[item].append(key)
            else:
                answer_question_map[item] = [key]

    return answer_question_map


if __name__ == '__main__':
    main()