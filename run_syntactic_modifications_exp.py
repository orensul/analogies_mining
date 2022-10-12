
import glob
import os
import runner
import json
import matplotlib.pyplot as plt
import sentence_bert
import nltk
import csv

from os.path import exists
from nltk.stem import WordNetLemmatizer
wn_lemma = WordNetLemmatizer()

propara_syntactic_modifications_exp_output_file = "propara_syntactic_modifications_exp"
folder_name = "../data/original_text_files/syntactic_wordtune_paraphrase_exp2/"
output_csv_file = "precision_k_automatic_paraphrases.csv"

mappings_models = [runner.FMV, runner.FMQ]
text_similarity_models = [runner.SBERT]
models_short_names = {runner.FMV: 'FMV', runner.FMQ: 'FMQ', runner.SBERT: 'SBERT'}
model_to_cos_sim_thresholds = runner.MODELS_SIM_THRESHOLD
k = 100
run_identical_verbs_stats = False


def run_exp(run_coref=False, run_qasrl=False, run_mappings=True, run_text_similarity=True):
    """
    Run the experiment -- exp3 (robustness -- automatic paraphrases)
    """
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
    if run_identical_verbs_stats:
        get_verbs_stats(pair_of_inputs)
    if run_mappings:
        for model_name in mappings_models:
            cos_sim_threshold = model_to_cos_sim_thresholds[model_name]
            run_propara_syntactic_modifications_exp_mappings(model_name, pair_of_inputs, cos_sim_threshold)

    if run_text_similarity:
        for model_name in text_similarity_models:
            run_propara_syntactic_modifications_exp_sent_bert(model_name, pair_of_inputs)

    result = []
    for model_name in mappings_models:
        cos_sim_threshold = model_to_cos_sim_thresholds[model_name]
        file_name = propara_syntactic_modifications_exp_output_file + "_model_" + model_name + "_cos_sim_" + str(cos_sim_threshold) + ".jsonl"
        result.append((file_name, model_name, cos_sim_threshold))

    for model_name in text_similarity_models:
        file_name = propara_syntactic_modifications_exp_output_file + "_model_" + model_name + ".jsonl"
        result.append((file_name, model_name, None))

    show_exp_results(result)



def run_propara_syntactic_modifications_exp_mappings(model_name, pair_of_inputs, cos_sim_threshold):
    """
    Run mappings (FMQ or FMV) on pairs of paragraphs from exp3 (robustness -- automatic paraphrases).
    Write into jsonl the ranked list of pair of paragraphs according to score.
    """
    saved_pairs_results = get_saved_pairs_results(model_name, cos_sim_threshold)

    pairs = runner.get_pair_of_inputs_qasrl_path(pair_of_inputs)
    count_pairs = 1
    results = []
    for pair in pairs:
        print(pair[0], pair[1])
        print("pair " + str(count_pairs) + " out of " + str(len(pairs)))
        path1, path2 = runner.extract_file_name_from_full_qasrl_path(pair[0]), \
                       runner.extract_file_name_from_full_qasrl_path(pair[1])
        count_pairs += 1
        if (path1, path2) in saved_pairs_results:
            score = saved_pairs_results[(path1, path2)]
            results.append([(path1, path2), score])
        else:
            solution, _, _ = runner.run_model(model_name, pair, cos_sim_threshold)
            score = 0 if not solution else runner.calc_solution_total_score(solution)
            results.append([(path1, path2), score])

    results.sort(key=lambda x: x[1], reverse=True)
    output_file_name = propara_syntactic_modifications_exp_output_file + "_model_" + model_name + "_cos_sim_" + str(cos_sim_threshold) + ".jsonl"
    with open(output_file_name, 'w') as output_file:
        json.dump(results, output_file)


def run_propara_syntactic_modifications_exp_sent_bert(model_name, pair_of_inputs):
    """
    Run SBERT on pairs of paragraphs from exp3 (robustness -- automatic paraphrases).
    Write into jsonl the ranked list of pair of paragraphs according to score (in the case of SBERT it is
    the cosine distance between the texts).
    """
    saved_pairs_results = get_saved_pairs_results(model_name)
    results = []
    for pair in pair_of_inputs:
        if (pair[0], pair[1]) in saved_pairs_results:
            score = saved_pairs_results[(pair[0], pair[1])]
            results.append([(pair[0], pair[1]), score])
        else:
            score = sentence_bert.get_text_similarity_score(pair)
            results.append([(pair[0], pair[1]), score])
    results.sort(key=lambda x: x[1], reverse=True)
    output_file_name = propara_syntactic_modifications_exp_output_file + "_model_" + model_name + ".jsonl"
    with open(output_file_name, 'w') as output_file:
        json.dump(results, output_file)


def get_verbs_stats(pair_of_inputs):
    count_identical_verbs, count_total_comparisons = 0, 0
    path_before = os.getcwd()
    os.chdir('./data/coref_text_files/syntactic_wordtune_paraphrase_exp')
    for pair in pair_of_inputs:
        s1, s2 = pair[0], pair[1]
        s1_pos = s1.find("_wordtune") if s1.find("_wordtune") != -1 else s1.find("_propara")
        s2_pos = s2.find("_wordtune") if s2.find("_wordtune") != -1 else s2.find("_propara")

        paragraph1, paragraph2 = s1[:s1_pos], s2[:s2_pos]
        if paragraph1 != paragraph2:
            continue

        f1_lines, f2_lines = read_lines_in_file(s1 + ".txt"), read_lines_in_file(s2 + ".txt")
        for i in range(len(f1_lines)):
            verbs_line_f1 = get_verbs_from_sentence(f1_lines[i])
            verbs_line_f2 = get_verbs_from_sentence(f2_lines[i])
            for v1 in verbs_line_f1:
                for v2 in verbs_line_f2:
                    if v1 == v2:
                        count_identical_verbs += 1
                    count_total_comparisons += 1

    percent_identical_verbs = round(count_identical_verbs / count_total_comparisons, 2)
    print("ratio identical verbs / total comparisons: " + str(percent_identical_verbs))
    os.chdir(path_before)

def get_saved_pairs_results(model_name, cos_sim_threshold=None):
    """
    Returns the saved pairs scores from existing output jsonl file (if exist) to avoid run FMQ / FMV mappings or SBERT
     on the pairs, again.
    """
    saved_pairs_results = {}
    if model_name in [runner.FMQ, runner.FMV]:
        curr_run_name = propara_syntactic_modifications_exp_output_file + "_model_" + model_name + "_cos_sim_" + str(cos_sim_threshold) + ".jsonl"
    else: #SBERT
        curr_run_name = propara_syntactic_modifications_exp_output_file + "_model_" + model_name + ".jsonl"

    if exists(curr_run_name):
        input_file = open(curr_run_name, 'r')
        for json_dict in input_file:
            json_object = json.loads(json_dict)
            for l in json_object:
                saved_pairs_results[(l[0][0], l[0][1])] = l[1]
    return saved_pairs_results


def get_verbs_from_sentence(sentence):
    tuples = preprocess(sentence)
    verbs = [tup[0] for tup in tuples if tup[1][0:2] == "VB"]
    lemma_verbs = [wn_lemma.lemmatize(v, 'v') for v in verbs]
    return [v for v in lemma_verbs if v != 'be']


def read_lines_in_file(filename):
    with open(filename) as f:
        return f.readlines()


def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent


def show_exp_results(result):
    """
    Write Precision 25 / 50 / 75 / 100 for every method into csv as well draw a full plot (X-axis is K from 1 to 100
    and Y-axis is the precision@K)
    """
    model_precisions = {}

    for output_file, model_name, cos_sim_threshold in result:
        scores_list, labels_list = create_scores_labels_from_results(output_file)
        print("Pos pairs: " + str(len([i for i in labels_list if i == 1])))
        print("Neg pairs: " + str(len([i for i in labels_list if i == 0])))
        tuples = list(zip(scores_list, labels_list))
        precisions = []

        for i in range(1, k+1):
            count_true = 0
            for j in range(i):
                label = tuples[j][1]
                if label == 1:
                    count_true += 1
            precision_i = round(count_true / i, 2)
            precisions.append(precision_i)
            if i % 25 == 0:
                print("precision " + str(i) + " " + model_name + " : " + str(precision_i))

        model_precisions[model_name] = precisions

    header = ['Method', 'P@25', 'P@50', 'P@75', 'P@100']

    fmq_precisions, fmv_precisions, sbert_precisions = model_precisions[runner.FMQ], model_precisions[runner.FMV], model_precisions[runner.SBERT]

    with open(output_csv_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow([models_short_names[runner.FMV], fmv_precisions[24], fmv_precisions[49], fmv_precisions[74],
                         fmv_precisions[99]])
        writer.writerow([models_short_names[runner.FMQ], fmq_precisions[24], fmq_precisions[49], fmq_precisions[74],
                         fmq_precisions[99]])
        writer.writerow([models_short_names[runner.SBERT], sbert_precisions[24], sbert_precisions[49], sbert_precisions[74],
                     sbert_precisions[99]])

    for model_name, precisions in model_precisions.items():
        x = [i for i in range(1, k+1)]
        y = precisions

        if model_name == runner.FMQ:
            label = models_short_names[runner.FMQ]
            plt.plot(x, y, label=label)
        elif model_name == runner.FMV:
            label = models_short_names[runner.FMV]
            plt.plot(x, y, label=label, linestyle='--')
        else:
            label = models_short_names[runner.SBERT]
            plt.plot(x, y, label=label, linestyle='-.')

    plt.ylim(0, 1.01)
    plt.xlim(1, 100)
    plt.xlabel('K')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()


def create_scores_labels_from_results(output_file_name):
    """
    Returns scores and labels lists of the pairs of paragraphs in this experiment.
    """
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


if __name__ == '__main__':
    run_exp(run_coref=False, run_qasrl=False, run_mappings=True, run_text_similarity=True)

