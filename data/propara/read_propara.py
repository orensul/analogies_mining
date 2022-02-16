
import json

import argparse


propara_train_tsv_file = 'grids.v1.train.tsv'
propara_train_json_file = 'grids.v1.train.json'
propara_dev_json_file = 'grids.v1.dev.json'
propara_test_json_file = 'grids.v1.test.json'
propara_train_para_id_title = 'grids.v1.train.para_id_title.json'
dataset_type_file = {"train": propara_train_json_file, "dev": propara_dev_json_file, "test": propara_test_json_file}

original_text_files_path = '../original_text_files/'

def main(args):
    paragraph_titles = get_paragraph_titles(propara_train_tsv_file)
    data = read_propara_paragraphs(dataset_type_file[args.dataset_type])
    para_id_title_map = {}
    for i in range(len(data)):
        para_id_title_map[data[i]["para_id"]] = paragraph_titles[i]


    write_paragraph_id_title(para_id_title_map, propara_train_para_id_title)
    converted_data = [{} for _ in range(len(data))]
    for idx, sample in enumerate(data):
        para_id, texts = sample["para_id"], sample["texts"]
        converted_data[idx]["para_id"], converted_data[idx]["texts"] = para_id, texts

    write_original_text_files(converted_data)


def write_paragraph_id_title(para_id_title_map, filename):
    with open(filename, 'w') as output_file:
        json.dump(para_id_title_map, output_file)



def get_paragraph_titles(filename):
    paragraph_titles = []
    f = open(filename, "r")
    lines = f.readlines()
    for line in lines:
        if "\t\tPROMPT:" not in line:
            continue
        start = line.find("\t\tPROMPT:") + len("\t\tPROMPT:")
        end = line.find("\t-")
        paragraph_title = line[start+1:end]
        paragraph_titles.append(paragraph_title)
    return paragraph_titles


def write_original_text_files(converted_data):
    for data in converted_data:
        para_id, texts = data['para_id'], data['texts']
        output_file_path = original_text_files_path + 'propara_para_id_' + para_id + '.txt'
        output_file = open(output_file_path, 'w')
        for line in texts:
            output_file.write(line)
            output_file.write("\n")



def read_propara_paragraphs(filename):
    f = open(filename, "r")
    lines = f.readlines()
    data = [{} for _ in range(len(lines))]
    for idx, line in enumerate(lines):
        json_object = json.loads(line)

        para_id, texts, participants, states = json_object['para_id'], json_object['sentence_texts'], \
                                               json_object['participants'], json_object['states']
        data[idx]['para_id'], data[idx]['texts'], data[idx]['participants'], data[idx]['states'] = para_id, texts, \
                                                                                                   participants, states
    return data




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, default='train',
                        help='possible values: train / dev / test')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
