
import glob
import os
import runner
import json
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import metrics

import sentence_bert

run_coref = False
run_qasrl = False
run_mappings = False
run_text_similarity = False

propara_syntactic_modifications_exp_output_file = "propara_syntactic_modifications_exp"
folder_name = "../data/original_text_files/syntactic_wordtune_paraphrase_exp2/"
mappings_models = ['findMappingsV', 'findMappingsQ']

model_to_cos_sim_thresholds = {'findMappingsV': [0.5], 'findMappingsQ': [0.7] , 'sentenceBert': []}
text_similarity_models = ["sentenceBert"]


def run_exp():
    os.chdir('s2e-coref')
    text_file_names = [f for f in glob.glob(folder_name + "*")]
    text_file_names = [f.replace(folder_name, "") for f in text_file_names]

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
    if run_mappings:
        for model_name in mappings_models:
            for cos_sim_threshold in model_to_cos_sim_thresholds[model_name]:
                run_propara_syntactic_modifications_exp_mappings(model_name, pair_of_inputs, cos_sim_threshold)

    if run_text_similarity:
        for model_name in text_similarity_models:
            run_propara_syntactic_modifications_exp_sent_bert(model_name, pair_of_inputs)

    result = []
    for model_name in text_similarity_models + mappings_models:
        for cos_sim_threshold in model_to_cos_sim_thresholds[model_name]:
            file_name = propara_syntactic_modifications_exp_output_file + "_model_" + model_name + "_cos_sim_" + str(cos_sim_threshold) + ".jsonl"
            result.append((file_name, model_name, cos_sim_threshold))

    for model_name in text_similarity_models:
        file_name = propara_syntactic_modifications_exp_output_file + "_model_" + model_name + ".jsonl"
        result.append((file_name, model_name, None))

    show_exp_results(result)


def run_propara_syntactic_modifications_exp_sent_bert(model_name, pair_of_inputs):
    results = []
    for pair in pair_of_inputs:
        score = sentence_bert.get_text_similarity_score(pair)
        results.append([(pair[0], pair[1]), score])
    results.sort(key=lambda x: x[1], reverse=True)
    output_file_name = propara_syntactic_modifications_exp_output_file + "_model_" + model_name + ".jsonl"
    with open(output_file_name, 'w') as output_file:
        json.dump(results, output_file)



def create_scores_labels_from_results(output_file_name):
    exp_pairs_rank = []
    input_file = open(output_file_name, 'r')
    for json_dict in input_file:
        json_object = json.loads(json_dict)
        for l in json_object:
            input1, input2, score = l[0][0], l[0][1], l[1]
            input1_prompt = input1.partition("_wordtune")[0] if "_wordtune" in input1 else \
                input1.partition("_propara")[0]
            input2_prompt = input2.partition("_wordtune")[0] if "_wordtune" in input2 else \
                input2.partition("_propara")[0]
            label = 1 if input1_prompt == input2_prompt else 0
            exp_pairs_rank.append((input1, input2, label, score))

    scores_list = [quad[-1] for quad in exp_pairs_rank]
    labels_list = [quad[2] for quad in exp_pairs_rank]
    return scores_list, labels_list


def show_exp_results(result):
    for output_file, model_name, cos_sim_threshold in result:
        scores_list, labels_list = create_scores_labels_from_results(output_file)
        fpr, tpr, thresholds = metrics.roc_curve(np.array(labels_list), np.array(scores_list))
        auc = round(metrics.auc(fpr, tpr), 3)
        print("AUC: ", auc)
        if cos_sim_threshold:
            label = "Method=" + model_name + ", AUC=" + str(auc)
        else:
            label = "Method=" + model_name + ", AUC=" + str(auc)
        plt.plot(fpr, tpr, label=label)

    plt.style.use('seaborn')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.show()


def run_propara_syntactic_modifications_exp_mappings(model_name, pair_of_inputs, cos_sim_threshold):
    pairs = runner.get_pair_of_inputs_qasrl_path(pair_of_inputs)
    count_pairs = 1
    results = []
    for pair in pairs:
        print(pair[0], pair[1])
        print("pair " + str(count_pairs) + " out of " + str(len(pairs)))
        path1, path2 = runner.extract_file_name_from_full_qasrl_path(pair[0]), \
                       runner.extract_file_name_from_full_qasrl_path(pair[1])
        count_pairs += 1
        solution, _, _ = runner.run_model(model_name, pair, cos_sim_threshold)
        score = 0 if not solution else runner.calc_solution_total_score(solution)
        print("pair: ", pair)
        print("score: ", score)
        results.append([(path1, path2), score])

    results.sort(key=lambda x: x[1], reverse=True)
    output_file_name = propara_syntactic_modifications_exp_output_file + "_model_" + model_name + "_cos_sim_" + str(cos_sim_threshold) + ".jsonl"
    with open(output_file_name, 'w') as output_file:
        json.dump(results, output_file)

if __name__ == '__main__':
    run_exp()