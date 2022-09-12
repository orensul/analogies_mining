import os, stat
import json
import en_core_web_sm

nlp = en_core_web_sm.load()

coref_predict_file_path = 'predict.py'
coref_input_file_path = 'input_files/input.jsonl'
text_files_dir = '../data/original_text_files'
coref_text_files_dir = '../data/coref_text_files'

replace_one_word_token_with_cluster_representative = True

pronouns = {"all", "another", "any", "anybody", "anyone", "anything", "as", "aught", "both", "each", "each other",
            "either", "enough", "everybody", "everyone", "everything", "few", "he", "her", "hers", "herself", "him",
            "himself", "his", "I", "idem", "it", "its", "itself", "many", "me", "mine", "most", "my", "myself",
            "naught", "neither", "no one", "nobody", "none", "nothing", "nought", "one", "one another", "other",
            "others", "ought", "our", "ours", "ourself", "ourselves", "several", "she", "some", "somebody", "someone",
            "something", "somewhat", "such", "suchlike", "that", "thee", "their", "theirs", "theirself", "theirselves",
            "them", "themself", "themselves", "there", "these", "they", "thine", "this", "those", "thou", "thy",
            "thyself", "us", "we", "what", "whatever", "whatnot", "whatsoever", "whence", "where", "whereby",
            "wherefrom", "wherein", "whereinto", "whereof", "whereon", "wherever", "wheresoever", "whereto",
            "whereunto", "wherewith", "wherewithal", "whether", "which", "whichever", "whichsoever", "who",
            "whoever", "whom", "whomever", "whomso", "whomsoever", "whose", "whosever", "whosesoever", "whoso",
            "whosoever", "ye", "yon", "yonder", "you", "your", "yours", "yourself", "yourselves"}


def main():
    create_coref_text_files(text_files_dir)


def create_coref_text_files(original_text_files):
    for text_input_file in original_text_files:
        text_input_file_path = text_files_dir + "/" + text_input_file
        create_coref_input_file(text_input_file_path, coref_input_file_path)
        os.chmod(coref_predict_file_path, stat.S_IRWXU)
        command = "python " + coref_predict_file_path + " --input_file " + coref_input_file_path
        os.system(command)
        sorted_tokens_range_result, tokens = read_coref_file(coref_input_file_path)
        write_coref_files(os.path.join(coref_text_files_dir, text_input_file), sorted_tokens_range_result, tokens)


def create_coref_input_file(text_input_file_path, coref_input_file_path):
    """
    create input file for coref by chunking sentences from the original text file
    """
    input_file = open(text_input_file_path, 'r')
    dictionary = {"tokens": []}
    for line in input_file:
        tokens = line.split(' ')
        for token in tokens:
            if not token:
                continue
            if ".\n" in token:
                token = token.replace(".\n", "")
                dictionary["tokens"].append(token)
                dictionary["tokens"].append(".")
                continue
            if token[-1] == ".":
                dictionary["tokens"].append(token[:-1])
                continue
            dictionary["tokens"].append(token)

    with open(coref_input_file_path, 'w') as output_file:
        json.dump(dictionary, output_file)


def read_coref_file(coref_input_file_path):
    """
    read the clusters from coref model result, choose representative and tokens to be replaced and return list of tuples
    with the range of indices of a token to be replaced and the range of indices of the representative, and the tokens themselves.
    """
    input_file = open(coref_input_file_path, 'r')
    tokens_range_result = []

    for json_dict in input_file:
        json_object = json.loads(json_dict)
        if "clusters" not in json_object:
            return [], json_object["tokens"]
        clusters, tokens = json_object["clusters"], json_object["tokens"]
        for cluster in clusters:
            lists_of_tokens = []
            for tokens_range in cluster:
                list_tokens = tokens[tokens_range[0]:tokens_range[1] + 1]
                lists_of_tokens.append(list_tokens)

            chosen_list_of_tokens = choose_list(lists_of_tokens)

            print("chosen list of tokens: ")
            print(chosen_list_of_tokens)
            if not chosen_list_of_tokens:
                continue

            chosen_tokens_range = list_to_tokens_range(chosen_list_of_tokens, cluster, tokens)

            for tokens_range in cluster:
                if replace_one_word_token_with_cluster_representative and tokens_range[1] > tokens_range[0]:
                    continue
                if tokens_range == chosen_tokens_range:
                    continue
                tokens_range_result.append((tokens_range, chosen_tokens_range))

    sorted_tokens_range_result = sorted(tokens_range_result, key=lambda x: x[0][0])
    return sorted_tokens_range_result, tokens


def write_coref_files(output_file_path, sorted_tokens_range_result, tokens):
    """
    Generate and write the text in the output coref file, which will be the input to the next phase -- QA-SRL.
    """
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
    output_str = output_str.replace(" .", ".\n")
    output_str = output_str.replace("\n ", "\n")
    output_file.write(output_str)


def choose_list(lists_of_tokens):
    """
    return the the representative of a cluster -- the shortest string (list of tokens) which is not a pronoun or a verb
    """
    print("list of tokens: ")
    print(lists_of_tokens)
    filtered_lists_of_tokens = filter_pronoun_tokens(lists_of_tokens)
    filtered_lists_of_tokens.sort(key=len)
    print("filtered sorted without duplicates list of tokens: ")
    print(filtered_lists_of_tokens)

    return filtered_lists_of_tokens[0] if len(filtered_lists_of_tokens) > 0 else None


def filter_pronoun_tokens(lists_of_tokens):
    """
    filter one-word token if it is a pronoun or a verb and remove duplicates
    """
    filtered_lists_of_tokens = []
    seen = set()
    for list_of_tokens in lists_of_tokens:
        list_of_tokens_str = " ".join(
            [token.lower() for token in [token[:-1] if token[-1] == ',' else token for token in list_of_tokens]])
        if len(list_of_tokens) > 1:
            if list_of_tokens_str not in seen:
                seen.add(list_of_tokens_str)
                filtered_lists_of_tokens.append(list_of_tokens)
        elif len(list_of_tokens) == 1:
            if list_of_tokens_str not in seen:
                seen.add(list_of_tokens_str)
                curr_list_of_tokens = list_of_tokens[0][:-1].lower() if list_of_tokens[0][-1] == ',' else \
                list_of_tokens[0].lower()
                if curr_list_of_tokens in pronouns or is_a_verb(curr_list_of_tokens):
                    continue
                filtered_lists_of_tokens.append(list_of_tokens)
    return filtered_lists_of_tokens


def is_a_verb(word):
    doc = nlp(word)
    return [token.pos_ for token in doc][0] == 'VERB'


def list_to_tokens_range(list_of_tokens, cluster, tokens):
    for tokens_range in cluster:
        if tokens[tokens_range[0]:tokens_range[1] + 1] == list_of_tokens:
            return tokens_range


if __name__ == '__main__':
    main()
