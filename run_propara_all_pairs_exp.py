import os
import glob

import runner
import json

import operator
from random import seed
from random import choice
import sentence_bert
import pandas as pd
import data.propara.read_propara as rp
from qa_srl import read_parsed_qasrl
from os.path import exists
run_coref = False
run_qasrl = False
run_mappings = False
run_random_samples = False
run_text_similarity = False

mappings_models = ['findMappingsV']
propara_results_path = "data/propara/propara_results"
propara_results_exp_format = "data/propara/propara_results_exp_format"
propara_map_path = './data/propara/grids.v1.train.para_id_title.json'


model_to_cos_sim_thresholds = {'findMappingsV': [0.5]}
text_similarity_models = ["sentenceBert"]

first_batch_from = 7
first_batch_till = 1500



def get_random_pairs_by_seed(pair_of_inputs, seed_val, random_sample_size):
    random_samples = []
    seed(seed_val)
    for _ in range(random_sample_size):
        random_samples.append(choice(pair_of_inputs))
    return random_samples


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

def run_exp():
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
            pair_of_inputs.append((file1, file2))

    os.chdir('../')
    # get_qasrl_questions_stats(pair_of_inputs)
    if run_mappings:
        for model_name in mappings_models:
            for cos_sim_threshold in model_to_cos_sim_thresholds[model_name]:
                run_propara_mappings(model_name, pair_of_inputs, cos_sim_threshold)

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
                pair[0]), runner.extract_file_name_from_full_qasrl_path(
                pair[1])
            prompt1, prompt2 = propara_para_id_title_map[path1.partition("para_id_")[2]], propara_para_id_title_map[
                path2.partition("para_id_")[2]]
            id1, id2 = path1[path1.rindex('_')+1:], path2[path2.rindex('_')+1:]

            pair_results.append((prompt1 + "(" + id1 + ")", prompt2 + "(" + id2 + ")", 0))

        propara_random_samples_path = "100_random_pairs.jsonl"
        with open(propara_random_samples_path, 'w') as output_file:
            json.dump(pair_results, output_file)

        format_propara_all_pairs_results(propara_random_samples_path, 'test', None)


def run_propara_sent_bert_similarity(model_name, pair_of_inputs):
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
    with open(propara_results_path_curr_run, 'w') as output_file:
        json.dump(pair_results, output_file)

    format_propara_all_pairs_results(propara_results_path_curr_run, model_name, None)

def read_propara_id_title_map(filename):
    input_file = open(filename, 'r')
    for json_dict in input_file:
        json_object = json.loads(json_dict)
        return json_object

def format_propara_all_pairs_results(propara_results_path, model_name, cos_sim_threshold):
    input_file = open(propara_results_path, 'r')
    if cos_sim_threshold is None:
        propara_results_exp_curr_run_format = propara_results_exp_format + "_model_" + model_name + ".xlsx"
    else:
        propara_results_exp_curr_run_format = propara_results_exp_format + "_model_" + model_name + "_cos_sim_" + str(cos_sim_threshold) + ".xlsx"

    writer = pd.ExcelWriter(propara_results_exp_curr_run_format, engine='xlsxwriter')
    writer.save()

    hash_table_result = {'BaseParagraphTopic': [], 'TargetParagraphTopic': [], 'Score': []}

    for json_dict in input_file:
        json_object = json.loads(json_dict)
        for row in json_object:
            hash_table_result['BaseParagraphTopic'].append(row[0])
            hash_table_result['TargetParagraphTopic'].append(row[1])
            hash_table_result['Score'].append(row[2])

    # dataframe Name and Age columns
    df = pd.DataFrame(hash_table_result)

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(propara_results_exp_curr_run_format, engine='xlsxwriter')

    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='Sheet1', index=False)

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()

    print(1)

def run_propara_mappings(model_name, pair_of_inputs, cos_sim_threshold):
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
    with open(propara_results_path_curr_run, 'w') as output_file:
        json.dump(pair_results, output_file)

    format_propara_all_pairs_results(propara_results_path_curr_run, model_name, cos_sim_threshold)


def get_num_words_from_paragraph_dict(d):
    l = 0
    texts = d['texts']
    for text in texts:
        l += len(text.split())
    return l

def get_paragraph_dict(propara_paragraphs, paragraph_id):
    for d in propara_paragraphs:
        if d['para_id'] == paragraph_id:
            return d

def read_results(file_path):
    input_file = open(file_path, 'r')
    pairs = []
    for json_dict in input_file:
        json_object = json.loads(json_dict)
        for l in json_object:
            id1 = l[0].split("(", 1)[1][:-1]
            id2 = l[1].split("(", 1)[1][:-1]
            pairs.append([id1, id2])
    result = []
    top_100_pairs = pairs[:100]
    propara_paragraphs = rp.read_propara_paragraphs('data/propara/grids.v1.train.json')
    for p1, p2 in top_100_pairs:
        d1 = get_paragraph_dict(propara_paragraphs, p1)
        p1_num_words = get_num_words_from_paragraph_dict(d1)
        d2 = get_paragraph_dict(propara_paragraphs, p2)
        p2_num_words = get_num_words_from_paragraph_dict(d2)
        result.append([p1, p2, p1_num_words, p2_num_words])
    write_paragraphs_num_words_results(result)

def write_paragraphs_num_words_results(result):
    writer = pd.ExcelWriter("paragraphs_num_words_results.xlsx", engine='xlsxwriter')
    writer.save()

    hash_table_result = {'P1': [], 'NumWordsP1': [], 'P2': [], 'NumWordsP2': []}


    for p1, p2, nw_p1, nw_p2 in result:
        hash_table_result['P1'].append(p1)
        hash_table_result['NumWordsP1'].append(str(nw_p1))
        hash_table_result['P2'].append(p2)
        hash_table_result['NumWordsP2'].append(str(nw_p2))


    # dataframe Name and Age columns
    df = pd.DataFrame(hash_table_result)

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter("paragraphs_num_words_results.xlsx", engine='xlsxwriter')

    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='Sheet1', index=False)

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()




if __name__ == '__main__':
    run_exp()
    read_results(propara_results_path + '.jsonl')