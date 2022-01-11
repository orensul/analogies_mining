

from coref import create_coref_text_files
from qa_srl import write_qasrl_output_files

text_files_dir = '../data/original_text_files'
coref_text_files_dir = '../data/coref_text_files'

def main():
    # create_coref_text_files(text_files_dir, coref_text_files_dir)
    write_qasrl_output_files(coref_text_files_dir)


if __name__ == '__main__':
    main()


