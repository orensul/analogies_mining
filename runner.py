
import os
import numpy as np
from coref import create_coref_text_files
from qa_srl import write_qasrl_output_files
from find_mappings import plot_bipartite_graph
import find_mappings
import find_mappings_verbs
from nltk.stem.porter import *
porterStemmer = PorterStemmer()


text_files_dir = '../data/original_text_files'
qasrl_prefix_path = './qasrl-modeling/data/'
qasrl_suffix_path = '_span_to_question.jsonl'


num_mappings_to_show = 7
default_cos_sim_threshold = 0.85
cos_sim_thresholds = [0.85]
top_k_for_medians_calc = 4

pipeline_config = {}

def run_pipeline(model_name, pair_of_inputs, run_coref, run_qasrl, run_mappings):
    os.chdir('s2e-coref')
    text_file_names = get_text_file_names()

    if run_coref:
        create_coref_text_files(text_file_names)

    os.chdir('../qasrl-modeling')
    if run_qasrl:
        write_qasrl_output_files(text_file_names)

    os.chdir('../')
    if run_mappings:
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        for pair in get_pair_of_inputs_qasrl_path(pair_of_inputs):
            solution1, solution2, solution3 = run_model(model_name, pair, default_cos_sim_threshold)
            if solution1 is None:
                continue

            plot_bipartite_graph(solution1[:num_mappings_to_show], colors[:num_mappings_to_show], default_cos_sim_threshold)

            if solution2:
                plot_bipartite_graph(solution2[:num_mappings_to_show], colors[:num_mappings_to_show],
                                 default_cos_sim_threshold)
            if solution3:
                plot_bipartite_graph(solution3[:num_mappings_to_show], colors[:num_mappings_to_show],
                                 default_cos_sim_threshold)



def run_model(model_name, pair, cos_sim_threshold):
    if model_name == "findMappingsQ":
        return find_mappings.generate_mappings(pair, cos_sim_threshold)
    elif model_name == "findMappingsV":
        return find_mappings_verbs.generate_mappings(pair, cos_sim_threshold)


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

# common functions to exp1 and exp3


def calc_solution_total_score(solution):
    """
    Returns the score of the solution for a pair of paragraphs. This score is used in experiment 1 -- mining analogies
    ranking formula: multiplying the number of mappings (in the solution) by the median similarity score
    (out of the similarity scores of the mappings in the solution). See Section 4.1 in the paper.
    """
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


if __name__ == '__main__':
    model_name = "findMappingsQ"
    pair_of_inputs = [('animal_cell', 'factory')]
    run_pipeline(model_name, pair_of_inputs, run_coref=False, run_qasrl=False, run_mappings=True)


