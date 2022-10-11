
import os
import glob
import runner
import json
import operator
import sentence_bert
import pandas as pd

from random import seed
from random import choice
from qa_srl import read_parsed_qasrl
from os.path import exists

propara_results_path = "data/propara/propara_results"
propara_results_exp_format = "data/propara/propara_results_exp_format"
propara_map_path = './data/propara/grids.v1.train.para_id_title.json'

mappings_models = [runner.FMQ, runner.FMV]
text_similarity_models = [runner.SBERT]
model_to_cos_sim_thresholds = runner.MODELS_SIM_THRESHOLD

first_batch_from = 7
first_batch_till = 1500


def run_exp(run_coref=False, run_qasrl=False, run_mappings=True, run_text_similarity=True, run_random_samples=False):
    """
    Run exp1 (analogies mining) on all pairs of paragraphs from ProPara, by running the different methods
    (FMQ, FMV, SBERT) and writing the results to xlsx. Can also run the results on 100 random pairs according to request
    (run_random_samples flag).
    """
    os.chdir('s2e-coref')
    text_file_names = [f for f in glob.glob("../data/original_text_files/propara_para_id_*")]
    text_file_names = [f.replace("../data/original_text_files/", "") for f in text_file_names]
    split_word = "propara_para_id_"
    text_file_names = [file_name for file_name in text_file_names if int(file_name.partition(split_word)[2][:-4]) if
                       int(file_name.partition(split_word)[2][:-4]) >= first_batch_from and int(
                           file_name.partition(split_word)[2][:-4]) <= first_batch_till and int(
                           file_name.partition(split_word)[2][:-4]) not in [159]]

    if run_coref:
        runner.create_coref_text_files(text_file_names)
    os.chdir('../qasrl-modeling')
    if run_qasrl:
        runner.write_qasrl_output_files(text_file_names)

    pair_of_inputs = []
    for i in range(len(text_file_names)):
        for j in range(i + 1, len(text_file_names)):
            file1, file2 = text_file_names[i].replace(".txt", ""), text_file_names[j].replace(".txt", "")
            pair_of_inputs.append((file1, file2))

    os.chdir('../')
    # get_qasrl_questions_stats(pair_of_inputs)
    if run_mappings:
        for model_name in mappings_models:
            run_propara_mappings(model_name, pair_of_inputs, model_to_cos_sim_thresholds[model_name])

    if run_text_similarity:
        for model_name in text_similarity_models:
            run_propara_sent_bert_similarity(model_name, pair_of_inputs)

    if run_random_samples:
        propara_para_id_title_map = read_propara_id_title_map(propara_map_path)
        random_pairs = get_random_pairs_by_seed(pair_of_inputs, 1234, 100)
        pairs = runner.get_pair_of_inputs_qasrl_path(random_pairs)
        pair_results = []
        for pair in pairs:
            path1, path2 = runner.extract_file_name_from_full_qasrl_path(
                pair[0]), runner.extract_file_name_from_full_qasrl_path(pair[1])

            prompt1, prompt2 = propara_para_id_title_map[path1.partition("para_id_")[2]], propara_para_id_title_map[
                path2.partition("para_id_")[2]]

            id1, id2 = path1[path1.rindex('_')+1:], path2[path2.rindex('_')+1:]
            pair_results.append((prompt1 + "(" + id1 + ")", prompt2 + "(" + id2 + ")", 0))

        propara_random_samples_path = "100_random_pairs.jsonl"
        with open(propara_random_samples_path, 'w') as output_file:
            json.dump(pair_results, output_file)

        format_propara_all_pairs_results(propara_random_samples_path, 'test', None)


def run_specific_paragraph_id_example(paragraph_id, run_coref=False, run_qasrl=False, run_mappings=True, run_text_similarity=True):
    """
    Run the mappings models (FMQ, FMV) and SBERT on all pairs of paragraphs from ProPara in which one of the paragraph is paragraph_id.
    Then, write the results to xlsx. It is used to analyze the results for a specific paragraph.
    """
    os.chdir('s2e-coref')
    text_file_names = [f for f in glob.glob("../data/original_text_files/propara_para_id_*")]
    text_file_names = [f.replace("../data/original_text_files/", "") for f in text_file_names]

    split_word = "propara_para_id_"
    text_file_names = [file_name for file_name in text_file_names if int(file_name.partition(split_word)[2][:-4]) if
                       int(file_name.partition(split_word)[2][:-4]) >= first_batch_from and int(
                           file_name.partition(split_word)[2][:-4]) <= first_batch_till and int(
                           file_name.partition(split_word)[2][:-4]) not in [159]]

    if run_coref:
        runner.create_coref_text_files(text_file_names)
    os.chdir('../qasrl-modeling')
    if run_qasrl:
        runner.write_qasrl_output_files(text_file_names)

    pair_of_inputs = []
    for i in range(len(text_file_names)):
        for j in range(i + 1, len(text_file_names)):
            file1 = text_file_names[i].replace(".txt", "")
            file2 = text_file_names[j].replace(".txt", "")
            if file1.endswith(paragraph_id) or file2.endswith(paragraph_id):
                pair_of_inputs.append((file1, file2))

    os.chdir('../')
    if run_mappings:
        for model_name in mappings_models:
            run_propara_mappings(model_name, pair_of_inputs, model_to_cos_sim_thresholds[model_name], paragraph_id)

    if run_text_similarity:
        for model_name in text_similarity_models:
            run_propara_sent_bert_similarity(model_name, pair_of_inputs, paragraph_id)


def read_propara_id_title_map(filename):
    input_file = open(filename, 'r')
    for json_dict in input_file:
        json_object = json.loads(json_dict)
        return json_object


def format_propara_all_pairs_results(propara_results_path, model_name, cos_sim_threshold, paragraph_id_suffix):
    """
    Write xlsx file with the base paragraph ID, target paragraph ID and the ranking score for pairs of paragraphs.
    This function is used for exp1 (analogies mining on all pairs of paragraphs) and for a specific demand of paragraph.
    """
    input_file = open(propara_results_path, 'r')
    if cos_sim_threshold is None:
        propara_results_exp_curr_run_format = propara_results_exp_format + "_model_" + model_name + paragraph_id_suffix + ".xlsx"
    else:
        propara_results_exp_curr_run_format = propara_results_exp_format + "_model_" + model_name + "_cos_sim_" + str(cos_sim_threshold) + paragraph_id_suffix + ".xlsx"

    hash_table_result = {'BaseParagraphTopic': [], 'TargetParagraphTopic': [], 'Score': []}

    for json_dict in input_file:
        json_object = json.loads(json_dict)
        for row in json_object:
            paragraph_id = paragraph_id_suffix.partition("_para_id_")[2]
            base_paragraph_id = row[0].partition("(")[2][:-1]
            # check if to create the results for specific example, so the chosen paragraph_id should be the base
            if not paragraph_id_suffix or base_paragraph_id == paragraph_id:
                hash_table_result['BaseParagraphTopic'].append(row[0])
                hash_table_result['TargetParagraphTopic'].append(row[1])
            else:
                hash_table_result['BaseParagraphTopic'].append(row[1])
                hash_table_result['TargetParagraphTopic'].append(row[0])
            hash_table_result['Score'].append(row[2])

    df = pd.DataFrame(hash_table_result)
    writer = pd.ExcelWriter(propara_results_exp_curr_run_format, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.save()


def get_random_pairs_by_seed(pair_of_inputs, seed_val, random_sample_size):
    random_samples = []
    seed(seed_val)
    for _ in range(random_sample_size):
        random_samples.append(choice(pair_of_inputs))
    return random_samples


def get_qasrl_questions_stats(pair_of_inputs):
    pairs = runner.get_pair_of_inputs_qasrl_path(pair_of_inputs)
    histogram = {}
    count_pairs = 0
    for pair in pairs:
        text1_answer_question_map = read_parsed_qasrl(pair[0])
        text2_answer_question_map = read_parsed_qasrl(pair[1])
        update_histogram_len_q_freq(text1_answer_question_map, histogram)
        update_histogram_len_q_freq(text2_answer_question_map, histogram)
        count_pairs += 1
        if count_pairs % 50 == 0:
            print("pair #" + str(count_pairs))
            percent_hist = get_percent_histogram(histogram)
            print(percent_hist)
    percent_hist = get_percent_histogram(histogram)
    print(percent_hist)


def update_histogram_len_q_freq(text_answer_question_map, histogram):
    for key, questions in text_answer_question_map.items():
        for question_tup in questions:
            question = question_tup[0]
            q_len = len(question.split(' '))
            if q_len not in histogram:
                histogram[q_len] = 0
            histogram[q_len] += 1


def get_percent_histogram(histogram):
    hist_percents = {}
    sum_freqs = sum(histogram.values())
    for len, freq in histogram.items():
        hist_percents[len] = round(freq / sum_freqs, 2)
    return hist_percents


def run_propara_mappings(model_name, pair_of_inputs, cos_sim_threshold, paragraph_id=None):
    """
    Run mappings (FMQ or FMV) on all pairs of paragraphs (read propara_results_path_curr_run file before to avoid
    running saved results). Write the results into xlsx file (base paragraph, target paragraph and ranking score).
    """
    pair_results = []
    propara_para_id_title_map = read_propara_id_title_map(propara_map_path)
    saved_pairs_results = {}
    propara_results_path_curr_run = propara_results_path + "_model_" + model_name + "_cos_sim_" + str(cos_sim_threshold) + ".jsonl"
    if exists(propara_results_path_curr_run):
        input_file = open(propara_results_path_curr_run, 'r')
        for json_dict in input_file:
            json_object = json.loads(json_dict)
            for l in json_object:
                saved_pairs_results[(l[0], l[1])] = l[2]

    pairs = runner.get_pair_of_inputs_qasrl_path(pair_of_inputs)
    count_pairs = 1
    for pair in pairs:
        print(pair[0], pair[1])
        print("pair " + str(count_pairs) + " out of " + str(len(pairs)))
        count_pairs += 1
        path1, path2 = runner.extract_file_name_from_full_qasrl_path(pair[0]), runner.extract_file_name_from_full_qasrl_path(
            pair[1])
        title1, title2 = propara_para_id_title_map[path1.partition("para_id_")[2]], propara_para_id_title_map[
            path2.partition("para_id_")[2]]

        if (title1 + "(" + path1.partition("para_id_")[2] + ")",
            title2 + "(" + path2.partition("para_id_")[2] + ")") in saved_pairs_results:
            pair_results.append((title1 + "(" + path1.partition("para_id_")[2] + ")",
                                 title2 + "(" + path2.partition("para_id_")[2] + ")", saved_pairs_results[(
                title1 + "(" + path1.partition("para_id_")[2] + ")",
                title2 + "(" + path2.partition("para_id_")[2] + ")")]))
            continue
        solution, _, _ = runner.run_model(model_name, pair, cos_sim_threshold)
        if not solution:
            pair_results.append((title1 + "(" + path1.partition("para_id_")[2] + ")",
                                 title2 + "(" + path2.partition("para_id_")[2] + ")", 0))
            continue
        score = runner.calc_solution_total_score(solution)
        pair_results.append((title1 + "(" + path1.partition("para_id_")[2] + ")",
                             title2 + "(" + path2.partition("para_id_")[2] + ")", score))

    pair_results = sorted(pair_results, key=operator.itemgetter(2), reverse=True)
    paragraph_id_suffix = "" if paragraph_id is None else "_para_id_" + paragraph_id
    propara_results_path_curr_run = propara_results_path_curr_run[:-6]
    propara_results_path_curr_run += paragraph_id_suffix + '.jsonl'
    with open(propara_results_path_curr_run, 'w') as output_file:
        json.dump(pair_results, output_file)

    format_propara_all_pairs_results(propara_results_path_curr_run, model_name, cos_sim_threshold, paragraph_id_suffix)


def run_propara_sent_bert_similarity(model_name, pair_of_inputs, paragraph_id=None):
    """
    Run SBERT on all pairs of paragraphs by cosine similarity (read propara_results_path_curr_run file before to avoid
    running saved results). Write the results into xlsx file (base paragraph, target paragraph and ranking score).
    """
    pair_results = []
    propara_para_id_title_map = read_propara_id_title_map(propara_map_path)
    saved_pairs_results = {}
    propara_results_path_curr_run = propara_results_path + "_model_" + model_name + ".jsonl"
    if exists(propara_results_path_curr_run):
        input_file = open(propara_results_path_curr_run, 'r')
        for json_dict in input_file:
            json_object = json.loads(json_dict)
            for l in json_object:
                saved_pairs_results[(l[0], l[1])] = l[2]

    count_pairs = 1
    for pair in pair_of_inputs:
        print(pair[0], pair[1])
        print("pair " + str(count_pairs) + " out of " + str(len(pair_of_inputs)))
        count_pairs += 1
        pair1, pair2 = pair
        title1, title2 = propara_para_id_title_map[pair1.partition("para_id_")[2]], propara_para_id_title_map[
            pair2.partition("para_id_")[2]]

        if (title1 + "(" + pair1.partition("para_id_")[2] + ")",
            title2 + "(" + pair2.partition("para_id_")[2] + ")") in saved_pairs_results:
            pair_results.append((title1 + "(" + pair1.partition("para_id_")[2] + ")",
                                 title2 + "(" + pair2.partition("para_id_")[2] + ")", saved_pairs_results[(
                title1 + "(" + pair1.partition("para_id_")[2] + ")",
                title2 + "(" + pair2.partition("para_id_")[2] + ")")]))
            continue

        score = sentence_bert.get_text_similarity_score(pair)
        pair_results.append((title1 + "(" + pair1.partition("para_id_")[2] + ")",
                             title2 + "(" + pair2.partition("para_id_")[2] + ")", score))

    pair_results = sorted(pair_results, key=operator.itemgetter(2), reverse=True)
    paragraph_id_suffix = "" if paragraph_id is None else "_para_id_" + paragraph_id
    propara_results_path_curr_run = propara_results_path_curr_run[:-6]
    propara_results_path_curr_run += paragraph_id_suffix + '.jsonl'

    with open(propara_results_path_curr_run, 'w') as output_file:
        json.dump(pair_results, output_file)

    format_propara_all_pairs_results(propara_results_path_curr_run, model_name, None, paragraph_id_suffix)


if __name__ == '__main__':
    run_exp(run_coref=False, run_qasrl=False, run_mappings=True, run_text_similarity=True, run_random_samples=False)

    specific_paragraph_id_examples = ['687', '779', '157']
    for paragraph_id_example in specific_paragraph_id_examples:
        run_specific_paragraph_id_example(paragraph_id_example, run_coref=False, run_qasrl=False, run_mappings=True, run_text_similarity=True)
