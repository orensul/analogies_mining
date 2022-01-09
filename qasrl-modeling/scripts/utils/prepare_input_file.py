
"""
This script reads a text file and converting it to jsonl file in the matching format
for qasrl pre-trained models.
The expected format of the input file for this script is:
<SENTENCE_ID>\t<SENTENCE>
Where each sentence should be in a different line.

Usage:
python scripts/utils/prepare_input_file.py --input data/input.txt --output data/output.jsonl
"""

from argparse import ArgumentParser
import json
import spacy
from lemminflect import getInflection, getLemma  # https://github.com/bjascob/LemmInflect


TAGS_FOR_LEMMINFLECT = {"past": "VBD",
                        "presentSingular3rd": "VBZ",
                        "presentParticiple": "VBG",
                        "pastParticiple": "VBN"}
SEPARATOR = '\t'


def convert_sentences(path_to_input, output_path):
    """
    converts an input file to a format that matches qasrl-modeling input file format.
    path_to_input is the path to the file to convert, and the expected format of the
    content in this file is "<SENTENCE_ID>\t<SENTENCE>\n" (Each line is a different
    sentence).
    """

    tokenizer = spacy.load("en_core_web_sm")

    with open(path_to_input, 'rt') as f:
        all_sentences = f.readlines()

    with open(output_path, 'wt') as out:
        for line in all_sentences:
            id_sentence, sentence_str = tuple(line.strip().split(SEPARATOR))
            tokens = tokenizer(sentence_str)
            line_dict = {"sentenceId": id_sentence, "sentenceTokens": [],
                         "verbEntries": {}}
            for i in range(len(tokens)):
                current_token_txt = tokens[i].text
                line_dict["sentenceTokens"].append(current_token_txt)
                if tokens[i].pos_ == "VERB":
                    inflect_verb(i, line_dict, current_token_txt)
            out.write(json.dumps(line_dict) + '\n')


def inflect_verb(verb_index, line_dict, current_token_txt):
    """
    builds a dictionary of the given verb inflections, and update line_dict with those
    inflections.
    """
    stem = getLemma(current_token_txt, upos='VERB')[0]
    inflected_forms = {"stem": stem}
    inflected_forms.update({k: getInflection(stem, tag=TAGS_FOR_LEMMINFLECT[k])[0]
                            for k in TAGS_FOR_LEMMINFLECT})
    line_dict["verbEntries"][str(verb_index)] = {"verbIndex": verb_index,
                                                 "verbInflectedForms": inflected_forms}


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', '-i', required=True,
                        help="path file to convert, each line represents a sentence "
                             "<ID>\\t<SENTENCE>")
    parser.add_argument('--output', '-o', required=True,
                        help="path to jsonl output, that will be used as an input for "
                             "the pre-trained model")
    args = parser.parse_args()

    convert_sentences(args.input, args.output)
