import os

from coref import create_coref_text_files
from qa_srl import write_qasrl_output_files
from find_mappings import generate_mappings

text_files_dir = '../data/original_text_files'
coref_text_files_dir = '../data/coref_text_files'

pair_of_inputs = [('./qasrl-modeling/data/animal_cell_span_to_question.jsonl',
                   './qasrl-modeling/data/factory_span_to_question.jsonl')]

run_coref = False
run_qasrl = False
run_mappings = True
def main():
    os.chdir('s2e-coref')
    if run_coref:
        create_coref_text_files(text_files_dir, coref_text_files_dir)
    os.chdir('../qasrl-modeling')
    if run_qasrl:
        write_qasrl_output_files(coref_text_files_dir)
    os.chdir('../')
    if run_mappings:
        for pair in pair_of_inputs:
            generate_mappings(pair)



if __name__ == '__main__':
    main()


