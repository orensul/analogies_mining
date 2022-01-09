import os, stat
import json
coref_predict_file_path = 's2e-coref/predict.py'
coref_input_file_path = 's2e-coref/input_files/input.jsonl'
text_input_file_path = 'data/animal_cell.txt'


def create_coref_input_file(text_input_file_path, coref_input_file_path):
    input_file = open(text_input_file_path, 'r')
    dictionary = {"tokens": []}
    for line in input_file:
        line = line.lower()
        tokens = line.split(' ')
        for token in tokens:
            if "\n" in token:
                token = token.replace("\n", "")
                token = token.replace(".", "")
                dictionary["tokens"].append(token)
                dictionary["tokens"].append(".")
                continue
            if token[-1] == ".":
                dictionary["tokens"].append(token[:-1])
                dictionary["tokens"].append(".")
                continue
            dictionary["tokens"].append(token)

    with open(coref_input_file_path, 'w') as output_file:
        json.dump(dictionary, output_file)





def main():
    create_coref_input_file(text_input_file_path, coref_input_file_path)
    os.chmod(coref_predict_file_path, stat.S_IRWXU)
    command = "python " + coref_predict_file_path + " --input_file " + coref_input_file_path
    os.system(command)



if __name__ == '__main__':
    main()

