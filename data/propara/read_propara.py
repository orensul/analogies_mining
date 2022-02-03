
import json

import argparse

propara_train_json_file = 'grids.v1.train.json'
propara_dev_json_file = 'grids.v1.dev.json'
propara_test_json_file = 'grids.v1.test.json'
dataset_type_file = {"train": propara_train_json_file, "dev": propara_dev_json_file, "test": propara_test_json_file}

original_text_files_path = '../original_text_files/'

def main(args):
    data = read_propara_paragraphs(dataset_type_file[args.dataset_type])
    converted_data = [{} for _ in range(len(data))]
    for idx, sample in enumerate(data):
        para_id, texts = sample["para_id"], sample["texts"]
        converted_data[idx]["para_id"], converted_data[idx]["texts"] = para_id, texts

    write_original_text_files(converted_data)


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
