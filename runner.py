import collections
import os
import glob
import json
from multiprocessing import Process
from time import sleep
import threading
import numpy as np
import operator
from coref import create_coref_text_files
from qa_srl import write_qasrl_output_files
from find_mappings import plot_bipartite_graph

import matplotlib.pyplot as plt
import find_mappings
import find_mappings_verbs
import pandas as pd

import nltk
from nltk.stem.porter import *
porterStemmer = PorterStemmer()

from sklearn.metrics import confusion_matrix

text_files_dir = '../data/original_text_files'
# pair_of_inputs = [('what_happens_during_photosynthesis1', 'what_happens_during_photosynthesis2'), ('animal_cell', 'factory'),
#                   ('electrical_circuit', 'water_pump'), ('digestion1', 'digestion2'), ('how_snow_forms1', 'how_snow_forms2')]




# pair_of_inputs = [('what_happens_during_photosynthesis1', 'what_happens_during_photosynthesis2')]
# pair_of_inputs = [('digestion1', 'digestion2')]
# pair_of_inputs = [('how_snow_forms1', 'how_snow_forms2')]

# good with coref take the shortest
# pair_of_inputs = [('animal_cell', 'factory')]

# good with coref take the shortest
# pair_of_inputs = [('electrical_circuit', 'water_pump')]


# pair_of_inputs = [('keane_general', 'keane_surgeon')]
# pair_of_inputs = [('rattermann_story1_base', 'rattermann_story1_target')]

# pair_of_inputs = [('rattermann_story2_base', 'rattermann_story2_target')]
# pair_of_inputs = [('rattermann_story3_base', 'rattermann_story3_target')]
# pair_of_inputs = [('rattermann_story4_base', 'rattermann_story4_target')]
# pair_of_inputs = [('rattermann_story5_base', 'rattermann_story5_target')]


# contains quotes and no analogies found
# pair_of_inputs = [('rattermann_story6_base', 'rattermann_story6_target')]

# pair_of_inputs = [('rattermann_story8_base', 'rattermann_story8_target')]


# good with coref take the shortest
# pair_of_inputs = [('rattermann_story9_base', 'rattermann_story9_target')]

# good with coref take the shortest
# pair_of_inputs = [('rattermann_story10_base', 'rattermann_story10_target')]



# good with coref take the shortest
# pair_of_inputs = [('rattermann_story11_base', 'rattermann_story11_target')]


# good with coref take the shortest
# pair_of_inputs = [('rattermann_story12_base', 'rattermann_story12_target')]

# pair_of_inputs = [('rattermann_story13_base', 'rattermann_story13_target')]


# good with coref take the shortest
# pair_of_inputs = [('rattermann_story14_base', 'rattermann_story14_target')]

# good with coref take the shortest
# pair_of_inputs = [('rattermann_story15_base', 'rattermann_story15_target')]

# pair_of_inputs = [('rattermann_story16_base', 'rattermann_story16_target')]

# pair_of_inputs = [('rattermann_story17_base', 'rattermann_story17_target')]


# good with coref take the shortest
# pair_of_inputs = [('rattermann_story18_base', 'rattermann_story18_target')]

# pair_of_inputs = [('test_base', 'test_target')]

pair_of_inputs = [('propara_para_id_627', 'propara_para_id_1217')]

propara_paraphrasing_exp_inputs = ['human_lifecycle', 'how_do_bats_use_echolocation', 'how_do_lungs_work', 'how_acid_rain_affect_env', 'process_of_recycling_aluminum_can']



# 'rattermann_story8_base_v1.txt', 'rattermann_story8_base_v2.txt',
# 'rattermann_story8_base_v3.txt', 'rattermann_story8_base_v4.txt',
# 'rattermann_story8_base_v5.txt', 'rattermann_story8_base_v6.txt',
# 'rattermann_story8_target_v1.txt', 'rattermann_story8_target_v2.txt',
# 'rattermann_story8_target_v3.txt', 'rattermann_story8_target_v4.txt',
# 'rattermann_story8_target_v5.txt', 'rattermann_story8_target_v6.txt',
#
# 'rattermann_story4_base_v1.txt', 'rattermann_story4_base_v2.txt',
# 'rattermann_story4_base_v3.txt', 'rattermann_story4_base_v4.txt',
# 'rattermann_story4_base_v5.txt', 'rattermann_story4_base_v6.txt',
# 'rattermann_story4_target_v1.txt', 'rattermann_story4_target_v2.txt',
# 'rattermann_story4_target_v3.txt', 'rattermann_story4_target_v4.txt',
# 'rattermann_story4_target_v5.txt', 'rattermann_story4_target_v6.txt',
#
# 'rattermann_story2_base_v1.txt', 'rattermann_story2_base_v2.txt',
# 'rattermann_story2_base_v3.txt', 'rattermann_story2_base_v4.txt',
# 'rattermann_story2_base_v5.txt', 'rattermann_story2_base_v6.txt',
# 'rattermann_story2_target_v1.txt', 'rattermann_story2_target_v2.txt',
# 'rattermann_story2_target_v3.txt', 'rattermann_story2_target_v4.txt',
# 'rattermann_story2_target_v5.txt', 'rattermann_story2_target_v6.txt',

stories_paraphrases = [
                        'rattermann_story2_base_v1.txt', 'rattermann_story2_base_v2.txt',
                        'rattermann_story2_base_v3.txt', 'rattermann_story2_base_v4.txt',
                        'rattermann_story2_base_v5.txt', 'rattermann_story2_base_v6.txt',
                        'rattermann_story2_target_v1.txt', 'rattermann_story2_target_v2.txt',
                        'rattermann_story2_target_v3.txt', 'rattermann_story2_target_v4.txt',
                        'rattermann_story2_target_v5.txt', 'rattermann_story2_target_v6.txt',

                        'rattermann_story12_base_v1.txt', 'rattermann_story12_base_v2.txt',
                        'rattermann_story12_base_v3.txt', 'rattermann_story12_base_v4.txt',
                        'rattermann_story12_base_v5.txt', 'rattermann_story12_base_v6.txt',
                        'rattermann_story12_target_v1.txt', 'rattermann_story12_target_v2.txt',
                        'rattermann_story12_target_v3.txt', 'rattermann_story12_target_v4.txt',
                        'rattermann_story12_target_v5.txt', 'rattermann_story12_target_v6.txt',



                       ]


propara_map_path = './data/propara/grids.v1.train.para_id_title.json'

propara_results_path = "data/propara/propara_results.jsonl"
propara_results_exp_format = "data/propara/propara_results_exp_format.xlsx"

qasrl_prefix_path = './qasrl-modeling/data/'
qasrl_suffix_path = '_span_to_question.jsonl'



run_coref = False
run_qasrl = False
run_mappings = True

generate_mappings_precision_oriented = True
num_mappings_to_show = 7
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
default_cos_sim_threshold = 0.85
cos_sim_thresholds = [0.75]
len_solution_cos_sim_threshold_map = {0: 0.7, 1: 0.75, 2: 0.8, 3: 0.8, 4: 0.85, 5: 0.85, 6: 0.85, 7: 0.85}
top_k_for_medians_calc = 4
first_batch_from = 7
first_batch_till = 1350

def calc_solution_total_score(solution):
    scores = [round(item[-1], 3) for item in solution[:top_k_for_medians_calc]]
    percentile_50 = np.percentile(scores, 50)
    if len(solution) == 0:
        return 0
    if len(solution) == 1:
        return percentile_50

    if percentile_50 > 1.0:
        return len(solution) * percentile_50
    return float(len(solution))

def extract_file_name_from_full_qasrl_path(path):
    path = path.replace(qasrl_prefix_path, "")
    path = path.replace("_span_to_question.jsonl", "")
    return path


def read_propara_id_title_map(filename):
    input_file = open(filename, 'r')
    for json_dict in input_file:
        json_object = json.loads(json_dict)
        return json_object












def get_identical_total_verbs_ratio(l):
    count_stemmed_identical_verbs = 0
    count_identical_verbs = 0
    for quad in l:
        v1, v2, v1_stem, v2_stem = quad
        if v1_stem == v2_stem:
            count_stemmed_identical_verbs += 1
        if v1 == v2:
            count_identical_verbs += 1
    identical_stemmed_verbs_out_of_total = count_stemmed_identical_verbs / len(l)
    identical_verbs_out_of_total = count_identical_verbs / len(l)

    return round(identical_verbs_out_of_total, 3), round(identical_stemmed_verbs_out_of_total, 3)



def run_propara_syntactic_exp(model_name, pair_of_inputs, cos_sim):
    pairs = get_pair_of_inputs_qasrl_path(pair_of_inputs)[:5]
    count_pairs = 1
    results = []
    list_of_verbs_to_compare = []
    for pair in pairs:
        print(pair[0], pair[1])
        print("pair " + str(count_pairs) + " out of " + str(len(pairs)))
        path1, path2 = extract_file_name_from_full_qasrl_path(pair[0]), extract_file_name_from_full_qasrl_path(
            pair[1])
        count_pairs += 1

        solution, _, _ = run_model(model_name, pair, default_cos_sim_threshold)
        if solution:
            for quad in solution:
                for mapping in quad[-2]:
                    v1, v2 = mapping[0][2], mapping[1][2]
                    v1_stem, v2_stem = porterStemmer.stem(v1), porterStemmer.stem(v2)
                    list_of_verbs_to_compare.append((v1, v2, v1_stem, v2_stem))
        score = 0 if not solution else calc_solution_total_score(solution)
        print("pair: ", pair)
        print("score: ", score)
        results.append([(path1, path2), score])

    results.sort(key=lambda x: x[1], reverse=True)
    with open(paraphrasing_syntactic_exp_output_file + "_cos_sim_" + str(cos_sim) + ".jsonl", 'w') as output_file:
        json.dump(results, output_file)

    print(get_identical_total_verbs_ratio(list_of_verbs_to_compare))

    stemmed_verbs_freq = collections.defaultdict(int)
    verbs_freq = collections.defaultdict(int)

    pair_of_verbs_freq = collections.defaultdict(int)

    for quad in list_of_verbs_to_compare:
        v1, v2, v1_stem, v2_stem = quad
        stemmed_verbs_freq[v1_stem] += 1
        stemmed_verbs_freq[v2_stem] += 1
        verbs_freq[v1] += 1
        verbs_freq[v2] += 1
        pair_of_verbs_freq[v1_stem + ":" + v2_stem] += 1

    print("verbs pairs freq")
    print({k: v for k, v in sorted(pair_of_verbs_freq.items(), key=lambda item: item[1])})

    print("stemmed verbs freq")
    print({k: v for k, v in sorted(stemmed_verbs_freq.items(), key=lambda item: item[1])})

    print("unique stemmed verbs: ")
    print(len(stemmed_verbs_freq.keys()))

    print("verbs freq")
    print({k: v for k, v in sorted(verbs_freq.items(), key=lambda item: item[1])})

    print("unique verbs: ")
    print(len(verbs_freq.keys()))


def run_propara_all_pairs_exp():
    os.chdir('s2e-coref')
    text_file_names = [f for f in glob.glob("../data/original_text_files/propara_para_id_*")]
    text_file_names = [f.replace("../data/original_text_files/", "") for f in text_file_names]

    split_word = "propara_para_id_"
    text_file_names = [file_name for file_name in text_file_names if int(file_name.partition(split_word)[2][:-4]) if int(file_name.partition(split_word)[2][:-4]) >= first_batch_from and int(file_name.partition(split_word)[2][:-4]) <= first_batch_till and int(file_name.partition(split_word)[2][:-4])  not in [159]]

    if run_coref:
        create_coref_text_files(text_file_names)
    os.chdir('../qasrl-modeling')
    if run_qasrl:
        write_qasrl_output_files(text_file_names)

    pair_of_inputs = []
    for i in range(len(text_file_names)):
        for j in range(i+1, len(text_file_names)):
            file1 = text_file_names[i].replace(".txt", "")
            file2 = text_file_names[j].replace(".txt", "")
            pair_of_inputs.append((file1, file2))

    os.chdir('../')
    if run_mappings:
        run_propara_mappings(pair_of_inputs)






def run_propara_mappings(pair_of_inputs):
    pair_results = []
    propara_para_id_title_map = read_propara_id_title_map(propara_map_path)
    saved_pairs_results = {}
    input_file = open(propara_results_path, 'r')
    for json_dict in input_file:
        json_object = json.loads(json_dict)
        for l in json_object:
            saved_pairs_results[(l[0], l[1])] = l[2]

    pairs = get_pair_of_inputs_qasrl_path(pair_of_inputs)
    count_pairs = 1
    for pair in pairs:
        print(pair[0], pair[1])
        print("pair " + str(count_pairs) + " out of " + str(len(pairs)))
        count_pairs += 1
        path1, path2 = extract_file_name_from_full_qasrl_path(pair[0]), extract_file_name_from_full_qasrl_path(
            pair[1])
        title1, title2 = propara_para_id_title_map[path1.partition("para_id_")[2]], propara_para_id_title_map[
            path2.partition("para_id_")[2]]

        if (title1 + "(" + path1.partition("para_id_")[2] + ")",
            title2 + "(" + path2.partition("para_id_")[2] + ")") in saved_pairs_results:
            pair_results.append((title1 + "(" + path1.partition("para_id_")[2] + ")",
                                 title2 + "(" + path2.partition("para_id_")[2] + ")", saved_pairs_results[(
            title1 + "(" + path1.partition("para_id_")[2] + ")", title2 + "(" + path2.partition("para_id_")[2] + ")")]))
            continue
        solution, _, _ = generate_mappings(pair, default_cos_sim_threshold)
        if not solution:
            pair_results.append((title1 + "(" + path1.partition("para_id_")[2] + ")",
                                 title2 + "(" + path2.partition("para_id_")[2] + ")", 0))
            continue
        if generate_mappings_precision_oriented or len(solution) >= num_mappings_to_show - 1:
            score = calc_solution_total_score(solution)
            pair_results.append((title1 + "(" + path1.partition("para_id_")[2] + ")",
                                 title2 + "(" + path2.partition("para_id_")[2] + ")", score))

    pair_results = sorted(pair_results, key=operator.itemgetter(2), reverse=True)
    with open(propara_results_path, 'w') as output_file:
        json.dump(pair_results, output_file)



def get_story_attributes_from_file_path(path):
    side = "base" if "_base_" in path else "target"
    start = path.find("story") + len("story")
    end = path.find("_base") if side == "base" else path.find("_target")
    num = path[start:end]
    start = path.find("base_") + len("base_") if side == "base" else path.find("target_") + len("target_")
    end = path.find("_span")
    version = path[start:end]
    return num, side, version


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    import itertools
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def run_stories_paraphrases():
    os.chdir('s2e-coref')
    text_file_names = stories_paraphrases

    if run_coref:
        create_coref_text_files(text_file_names)

    os.chdir('../qasrl-modeling')
    if run_qasrl:
        write_qasrl_output_files(text_file_names)

    pair_of_inputs = []
    for i in range(len(text_file_names)):
        for j in range(i + 1, len(text_file_names)):
            file1 = text_file_names[i].replace(".txt", "")
            file2 = text_file_names[j].replace(".txt", "")
            pair_of_inputs.append((file1, file2))

    os.chdir('../')
    if run_mappings:
        preds = []
        labels = []
        results = []
        for pair in get_pair_of_inputs_qasrl_path(pair_of_inputs):
            s1_num, s1_side, s1_version = get_story_attributes_from_file_path(pair[0])
            s2_num, s2_side, s2_version = get_story_attributes_from_file_path(pair[1])
            if s1_num == s2_num:
                labels.append(1)
                continue
            else:
                labels.append(0)
            print(pair)
            solution, _, _ = generate_mappings(pair, default_cos_sim_threshold)
            if not solution:
                results.append((s1_num + "_" + s1_side + "_" + s1_version + "_vs_" + s2_num + "_" + s2_side + "_" + s2_version, []))
                continue
            scores = calc_solution_total_score(solution)
            results.append((s1_num + "_" + s1_side + "_" + s1_version + "_vs_" + s2_num + "_" + s2_side + "_" + s2_version, scores))


        for result in results:
            print(result)
        # max_len_scores = 0
        # for tup in results:
        #     max_len_scores = max(len(tup[1]), max_len_scores)
        #
        #
        # feature_cols = []
        # columns = []
        # for i in range(max_len_scores):
        #     columns.append("score" + str(i+1))
        #     feature_cols.append("score" + str(i+1))
        # columns.append("label")
        #
        # scores_zero_padding = []
        # for tup in results:
        #     curr_scores_zero_padding = []
        #     for score in tup[1]:
        #         curr_scores_zero_padding.append(score)
        #     for i in range(len(curr_scores_zero_padding), max_len_scores):
        #         curr_scores_zero_padding.append(0)
        #     scores_zero_padding.append(curr_scores_zero_padding)
        #
        # list = [scores_zero_padding[i] + [labels[i]] for i in range(len(scores_zero_padding))]
        #
        #
        # df = pd.DataFrame(list, columns=columns)
        #
        #
        # X = df[feature_cols]  # Features
        # y = df.label  # Target variable
        #
        # from sklearn.model_selection import train_test_split
        # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)
        #
        # # import the class
        # from sklearn.linear_model import LogisticRegression
        #
        # # instantiate the model (using the default parameters)
        # logreg = LogisticRegression()
        #
        # # fit the model with data
        # logreg.fit(X_train, y_train)
        #
        # y_pred = logreg.predict(X_test)
        #
        # from sklearn import metrics
        # confusion = confusion_matrix(y_test, y_pred)
        # print('Confusion Matrix\n')
        # print(confusion)
        #
        # # importing accuracy_score, precision_score, recall_score, f1_score
        # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        # acc = accuracy_score(y_test, y_pred)
        # print('\nAccuracy: {:.2f}\n'.format(acc))
        #
        # micro_f1_score = f1_score(y_test, y_pred, average='micro')
        # print('Micro F1-score: {:.2f}\n'.format(micro_f1_score))
        # macro_f1_score = f1_score(labels, preds, average='macro')
        # print('Macro F1-score: {:.2f}\n'.format(macro_f1_score))
        #
        # cm = metrics.confusion_matrix(y_test, y_pred)
        # plot_confusion_matrix(cm, classes=['false', 'true'])
        # print('\nClassification Report\n')
        # print(classification_report(y_test, y_pred))
        # print(1)




def run_model(model_name, pair, cos_sim_threshold):
    if model_name == "findMappingsQ":
        return find_mappings.generate_mappings(pair, cos_sim_threshold)
    elif model_name == "findMappingsV":
        return find_mappings_verbs.generate_mappings(pair, cos_sim_threshold)


def main(model_name):
    os.chdir('s2e-coref')
    text_file_names = get_text_file_names()

    if run_coref:
        create_coref_text_files(text_file_names)

    os.chdir('../qasrl-modeling')
    if run_qasrl:
        write_qasrl_output_files(text_file_names)

    os.chdir('../')
    if run_mappings:
        for pair in get_pair_of_inputs_qasrl_path(pair_of_inputs):
            solution1, solution2, solution3 = run_model(model_name, pair, default_cos_sim_threshold)
            if solution1 is None:
                continue
            if generate_mappings_precision_oriented or len(solution1) >= num_mappings_to_show - 1:
                if solution1:
                    plot_bipartite_graph(solution1[:num_mappings_to_show], colors[:num_mappings_to_show], default_cos_sim_threshold)
                if solution2:
                    plot_bipartite_graph(solution2[:num_mappings_to_show], colors[:num_mappings_to_show],
                                     default_cos_sim_threshold)
                if solution3:
                    plot_bipartite_graph(solution3[:num_mappings_to_show], colors[:num_mappings_to_show],
                                     default_cos_sim_threshold)

                continue
            cos_sim_threshold = len_solution_cos_sim_threshold_map[len(solution1)]
            solution1, solution2, solution3 = run_model(model_name, pair, default_cos_sim_threshold)
            if solution1:
                plot_bipartite_graph(solution1[:num_mappings_to_show], colors[:num_mappings_to_show], cos_sim_threshold)
            if solution2:
                plot_bipartite_graph(solution2[:num_mappings_to_show], colors[:num_mappings_to_show], cos_sim_threshold)
            if solution3:
                plot_bipartite_graph(solution3[:num_mappings_to_show], colors[:num_mappings_to_show], cos_sim_threshold)


def get_pair_of_inputs_qasrl_path(pair_of_inputs):
    pairs_qasrl_path = []
    for pair in pair_of_inputs:
        new_pair = (qasrl_prefix_path + pair[0] + qasrl_suffix_path, qasrl_prefix_path  + pair[1] + qasrl_suffix_path)
        pairs_qasrl_path.append(new_pair)
    return pairs_qasrl_path

def get_text_file_names():
    text_files_path = []
    for pair in pair_of_inputs:
        text_files_path.append(pair[0] + ".txt")
        text_files_path.append(pair[1] + ".txt")
    return text_files_path


def format_propara_all_pairs_results(propara_results_path):
    input_file = open(propara_results_path, 'r')

    import pandas as pd
    writer = pd.ExcelWriter(propara_results_exp_format, engine='xlsxwriter')
    writer.save()
    import pandas as pd

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
    writer = pd.ExcelWriter(propara_results_exp_format, engine='xlsxwriter')

    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='Sheet1', index=False)

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()






if __name__ == '__main__':
    # format_propara_all_pairs_results(propara_results_path)
    # model_name = "find_mappings"
    model_name = "findMappingsQ"
    main(model_name)
    # run_propara_all_pairs_exp()
    # run_stories_paraphrases()


