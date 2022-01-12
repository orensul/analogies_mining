import os

from coref import create_coref_text_files
from qa_srl import write_qasrl_output_files
from find_mappings import generate_mappings

text_files_dir = '../data/original_text_files'
pair_of_inputs = [('digestion1', 'digestion2')]
# pair_of_inputs = [('animal_cell', 'factory')]

qasrl_prefix_path = './qasrl-modeling/data/'
qasrl_suffix_path = '_span_to_question.jsonl'

run_coref = False
run_qasrl = False
run_mappings = True


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
            generate_mappings(pair)


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


