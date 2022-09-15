import collections
import os
import glob
import json
import numpy as np
import operator
from coref import create_coref_text_files
from qa_srl import write_qasrl_output_files
from find_mappings import plot_bipartite_graph
import matplotlib.pyplot as plt
import find_mappings
import find_mappings_verbs
from nltk.stem.porter import *
porterStemmer = PorterStemmer()
text_files_dir = '../data/original_text_files'

# pair_of_inputs = [('what_happens_during_photosynthesis1', 'what_happens_during_photosynthesis2'), ('animal_cell', 'factory'),
#                   ('electrical_circuit', 'water_pump'), ('digestion1', 'digestion2'), ('how_snow_forms1', 'how_snow_forms2')]




# pair_of_inputs = [('what_happens_during_photosynthesis1', 'what_happens_during_photosynthesis2')]
# pair_of_inputs = [('digestion1', 'digestion2')]
# pair_of_inputs = [('how_snow_forms1', 'how_snow_forms2')]

# good with coref take the shortest
pair_of_inputs = [('rattermann_story2_base', 'rattermann_story2_target')]

# good with coref take the shortest
# pair_of_inputs = [('electrical_circuit', 'water_pump')]


# stories_mapping_eval = [('keane_general', 'keane_surgeon'),
#            ('rattermann_story1_base', 'rattermann_story1_target'),
#            ('rattermann_story2_base', 'rattermann_story2_target'),
#            ('rattermann_story3_base', 'rattermann_story3_target'),
#            ('rattermann_story4_base', 'rattermann_story4_target'),
#            ('rattermann_story5_base', 'rattermann_story5_target'),
#            ('rattermann_story6_base', 'rattermann_story6_target'),
#            ('rattermann_story8_base', 'rattermann_story8_target'),
#            ('rattermann_story9_base', 'rattermann_story9_target'),
#            ('rattermann_story11_base', 'rattermann_story11_target'),
#            ('rattermann_story12_base', 'rattermann_story12_target'),
#            ('rattermann_story13_base', 'rattermann_story13_target'),
#            ('rattermann_story14_base', 'rattermann_story14_target'),
#            ('rattermann_story15_base', 'rattermann_story15_target'),
#            ('rattermann_story16_base', 'rattermann_story16_target'),
#            ('rattermann_story17_base', 'rattermann_story17_target'),
#            ('rattermann_story18_base', 'rattermann_story18_target'),
#
#            ]

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
# pair_of_inputs = [('rattermann_story18_base', 'rattermann_story18_target')]

# propara_mappings_eval = [('propara_para_id_490', 'propara_para_id_1158'),
#                          ('propara_para_id_552', 'propara_para_id_626'),
#                          ('propara_para_id_524', 'propara_para_id_645'),
#                          ('propara_para_id_393', 'propara_para_id_392'),
#                          ('propara_para_id_587', 'propara_para_id_588'),
#                          ('propara_para_id_1291', 'propara_para_id_1014'),
#                          ('propara_para_id_330', 'propara_para_id_938'),
#                          ('propara_para_id_779', 'propara_para_id_938'),
#                          ('propara_para_id_157', 'propara_para_id_882'),
#                          ('propara_para_id_779', 'propara_para_id_330'),
#                          ('propara_para_id_644', 'propara_para_id_528'),
#                          ('propara_para_id_1224', 'propara_para_id_687'),
#                          ('propara_para_id_1127', 'propara_para_id_7'),
#                          ('propara_para_id_1158', 'propara_para_id_315')
#                          ]
# pair_of_inputs = [('propara_para_id_330', 'propara_para_id_938')]


propara_map_path = './data/propara/grids.v1.train.para_id_title.json'

propara_results_path = "data/propara/propara_results.jsonl"

qasrl_prefix_path = './qasrl-modeling/data/'
qasrl_suffix_path = '_span_to_question.jsonl'



run_coref = False
run_qasrl = False
run_mappings = True

generate_mappings_precision_oriented = True
num_mappings_to_show = 7
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
default_cos_sim_threshold = 0.5
cos_sim_thresholds = [0.7]
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


if __name__ == '__main__':
    model_name = "findMappingsV"
    main(model_name)


