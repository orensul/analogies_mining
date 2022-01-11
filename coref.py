import os, stat
import json
coref_predict_file_path = 'predict.py'
coref_input_file_path = 'input_files/input.jsonl'
text_files_dir = '../data/original_text_files'
coref_text_files_dir = '../data/coref_text_files'


def main():
    create_coref_text_files(text_files_dir, coref_text_files_dir)


def create_coref_text_files(text_files_dir, coref_text_files_dir):
    for text_input_file_path in os.listdir(text_files_dir):
        create_coref_input_file(os.path.join(text_files_dir, text_input_file_path), coref_input_file_path)
        os.chmod(coref_predict_file_path, stat.S_IRWXU)
        command = "python " + coref_predict_file_path + " --input_file " + coref_input_file_path
        os.system(command)
        sorted_tokens_range_result, tokens = read_coref_file(coref_input_file_path)
        write_coref_files(os.path.join(coref_text_files_dir, text_input_file_path),
                                  sorted_tokens_range_result, tokens)




def choose_list(lists_of_tokens):
    lists_of_tokens.sort(key=len, reverse=True)
    chosen_list = lists_of_tokens[0]
    if len(lists_of_tokens[1]) > 1:
        chosen_list = lists_of_tokens[1]
    return chosen_list


def read_coref_file(coref_input_file_path):
    input_file = open(coref_input_file_path, 'r')
    tokens_range_result = []
    for json_dict in input_file:
        json_object = json.loads(json_dict)
        clusters, tokens = json_object["clusters"], json_object["tokens"]
        for cluster in clusters:
            lists_of_tokens = []
            for tokens_range in cluster:
                list_tokens = tokens[tokens_range[0]:tokens_range[1]+1]
                lists_of_tokens.append(list_tokens)
            chosen_list_of_tokens = choose_list(lists_of_tokens)
            chosen_tokens_range = list_to_tokens_range(chosen_list_of_tokens, cluster, tokens)
            for tokens_range in cluster:
                if tokens_range == chosen_tokens_range:
                    continue
                tokens_range_result.append((tokens_range, chosen_tokens_range))

    sorted_tokens_range_result = sorted(tokens_range_result, key=lambda x: x[0][0])
    return sorted_tokens_range_result, tokens


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


def list_to_tokens_range(list_of_tokens, cluster, tokens):
    for tokens_range in cluster:
        if tokens[tokens_range[0]:tokens_range[1]+1] == list_of_tokens:
            return tokens_range


def write_coref_files(output_file_path, sorted_tokens_range_result, tokens):
    output_file = open(output_file_path, 'w')
    output = []
    curr_idx = 0
    for tokens_range in sorted_tokens_range_result:
        if tokens_range[0][0] <= curr_idx:
            continue
        for i in range(curr_idx, tokens_range[0][0]):
            output.append(tokens[i])
        curr_idx = tokens_range[0][1] + 1
        for i in range(tokens_range[1][0], tokens_range[1][1] + 1):
            output.append(tokens[i])
    for i in range(curr_idx, len(tokens)):
        output.append(tokens[i])

    output_str = " ".join(output)
    output_str = output_str.replace(" . ", ".\n")
    output_file.write(output_str)




if __name__ == '__main__':
    main()


