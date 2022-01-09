""" Usage:
    <file-name> --in=IN_FILE --out=OUT_FILE [--debug]
"""
# External imports
import logging
import pdb
from pprint import pprint
from pprint import pformat
from docopt import docopt
from collections import defaultdict
from operator import itemgetter
from tqdm import tqdm
import json
import _jsonnet

# Local imports

#=-----

bert_token_indexers = {
    "bert": {
        "type": "bert-pretrained",
        "pretrained_model": "bert-base-uncased",
        "do_lowercase": True,
        "use_starting_offsets": True
    },
    "token_characters": {
        "type": "characters",
        "min_padding_length": 3
    }
}

bert_text_field_embedder = {
    "allow_unmatched_keys": True,
    "embedder_to_indexer_map": {
        "bert": ["bert", "bert-offsets"],
        "token_characters": ["token_characters"],
    },
    "token_embedders": {
        "bert": {
            "type": "bert-pretrained",
            "pretrained_model": "bert-base-uncased" 
        }
    }
}


if __name__ == "__main__":
    # Parse command line arguments
    args = docopt(__doc__)
    inp_fn = args["--in"]
    out_fn = args["--out"]
    debug = args["--debug"]
    if debug:
        logging.basicConfig(level = logging.DEBUG)
    else:
        logging.basicConfig(level = logging.INFO)

    cur_config = json.loads(_jsonnet.evaluate_file(inp_fn))
    cur_config["dataset_reader"]["token_indexers"] = bert_token_indexers
    cur_config["model"]["sentence_encoder"]["text_field_embedder"] = bert_text_field_embedder
    cur_config["model"]["sentence_encoder"]["stacked_encoder"]["input_size"] = 868

    with open(out_fn, "w", encoding = "utf8") as fout:
        json.dump(cur_config, fout, indent=4, sort_keys=True)
        
    logging.info("DONE")
