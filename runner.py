import os

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

pair_of_inputs = [('test_base', 'test_target')]

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
        for pair in get_pair_of_inputs_qasrl_path():
            solution = generate_mappings(pair, default_cos_sim_threshold)
            if more_precision_oriented or len(solution) >= num_mappings_to_show - 1:
                plot_bipartite_graph(solution[:num_mappings_to_show], colors[:num_mappings_to_show], default_cos_sim_threshold)
                continue
            cos_sim_threshold = len_solution_cos_sim_threshold_map[len(solution)]
            solution = generate_mappings(pair, cos_sim_threshold)
            plot_bipartite_graph(solution[:num_mappings_to_show], colors[:num_mappings_to_show], cos_sim_threshold)





def get_pair_of_inputs_qasrl_path():
    pairs_qasrl_path = []
    for pair in pair_of_inputs:
        new_pair = (qasrl_prefix_path + pair[0] + qasrl_suffix_path, qasrl_prefix_path + pair[1] + qasrl_suffix_path)
        pairs_qasrl_path.append(new_pair)
    return pairs_qasrl_path

def get_text_file_names():
    text_files_path = []
    for pair in pair_of_inputs:
        text_files_path.append(pair[0] + ".txt")
        text_files_path.append(pair[1] + ".txt")
    return text_files_path

if __name__ == '__main__':
    main()


