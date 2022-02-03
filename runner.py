import os
import glob
import json
from coref import create_coref_text_files
from qa_srl import write_qasrl_output_files
from find_mappings import generate_mappings
from find_mappings import plot_bipartite_graph

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


# good with coref take the shortest
# pair_of_inputs = [('keane_general', 'keane_surgeon')]

# good with coref take the shortest
# pair_of_inputs = [('rattermann_story2_base', 'rattermann_story2_target')]

# pair_of_inputs = [('rattermann_story3_base', 'rattermann_story3_target')]


# good with coref take the shortest
# pair_of_inputs = [('rattermann_story4_base', 'rattermann_story4_target')]

# good with coref take the shortest
# pair_of_inputs = [('rattermann_story8_base', 'rattermann_story8_target')]

# good with coref take the shortest
# pair_of_inputs = [('rattermann_story12_base', 'rattermann_story12_target')]

# good with coref take the shortest
# pair_of_inputs = [('rattermann_story14_base', 'rattermann_story14_target')]

# good with coref take the shortest
# pair_of_inputs = [('rattermann_story16_base', 'rattermann_story16_target')]

# good with coref take the shortest
# pair_of_inputs = [('rattermann_story18_base', 'rattermann_story18_target')]

# pair_of_inputs = [('test_base', 'test_target')]

pair_of_inputs = [('propara_para_id_872', 'propara_para_id_687')]

qasrl_prefix_path = './qasrl-modeling/data/'
qasrl_suffix_path = '_span_to_question.jsonl'

run_coref = False
run_qasrl = False
run_mappings = True
more_precision_oriented = False
num_mappings_to_show = 5
colors = ['r', 'g', 'b', 'y', 'm']
default_cos_sim_threshold = 0.85
len_solution_cos_sim_threshold_map = {0: 0.65, 1: 0.75, 2: 0.8, 3: 0.8, 4: 0.85, 5: 0.85}

first_batch_from = 7
first_batch_till = 700

def calc_solution_total_score(solution):
    total_score = 0
    for tup in solution:
        total_score += tup[-1]
    return total_score

def extract_file_name_from_full_qasrl_path(path):
    path = path.replace(qasrl_prefix_path, "")
    path = path.replace("_span_to_question.jsonl", "")
    return path


def run_propara():
    propara_results_path = "data/propara/propara_results.jsonl"
    os.chdir('s2e-coref')
    text_file_names = [f for f in glob.glob("../data/original_text_files/propara_para_id_*")]
    text_file_names = [f.replace("../data/original_text_files/", "") for f in text_file_names]

    split_word = "propara_para_id_"
    text_file_names = [file_name for file_name in text_file_names if int(file_name.partition(split_word)[2][:-4]) >= first_batch_from and int(file_name.partition(split_word)[2][:-4]) <= first_batch_till]

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

    pair_results = []
    os.chdir('../')
    if run_mappings:
        saved_pairs_results = {}
        input_file = open(propara_results_path, 'r')
        for json_dict in input_file:
            json_object = json.loads(json_dict)
            for l in json_object:
                saved_pairs_results[(l[0], l[1])] = l[2]


        count = 1
        pairs = get_pair_of_inputs_qasrl_path(pair_of_inputs)
        for pair in pairs:
            path1, path2 = extract_file_name_from_full_qasrl_path(pair[0]), extract_file_name_from_full_qasrl_path(
                pair[1])
            print("pair " + str(count) + " out of " + str(len(pairs)) + ": path1=" + path1 + " path2=" + path2)
            count += 1
            if (path1, path2) in saved_pairs_results:
                pair_results.append((path1, path2, saved_pairs_results[(path1, path2)]))
                continue
            solution = generate_mappings(pair, default_cos_sim_threshold)
            if not solution:
                pair_results.append((path1, path2, 0))
                continue
            if more_precision_oriented or len(solution) >= num_mappings_to_show - 1:
                score = calc_solution_total_score(solution)
                pair_results.append((path1, path2, score))
                # plot_bipartite_graph(solution[:num_mappings_to_show], colors[:num_mappings_to_show], default_cos_sim_threshold)
                continue
            cos_sim_threshold = len_solution_cos_sim_threshold_map[len(solution)]
            solution = generate_mappings(pair, cos_sim_threshold)
            if not solution:
                pair_results.append((path1, path2, 0))
                continue
            score = calc_solution_total_score(solution)
            pair_results.append((path1, path2, score))
            # plot_bipartite_graph(solution[:num_mappings_to_show], colors[:num_mappings_to_show], cos_sim_threshold)

    pair_results.sort(key=lambda x: x[2], reverse=True)

    with open(propara_results_path, 'w') as output_file:
        json.dump(pair_results, output_file)


def main():
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
            solution = generate_mappings(pair, default_cos_sim_threshold)
            if more_precision_oriented or len(solution) >= num_mappings_to_show - 1:
                plot_bipartite_graph(solution[:num_mappings_to_show], colors[:num_mappings_to_show], default_cos_sim_threshold)
                continue
            cos_sim_threshold = len_solution_cos_sim_threshold_map[len(solution)]
            solution = generate_mappings(pair, cos_sim_threshold)
            plot_bipartite_graph(solution[:num_mappings_to_show], colors[:num_mappings_to_show], cos_sim_threshold)





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
    # main()
    run_propara()


