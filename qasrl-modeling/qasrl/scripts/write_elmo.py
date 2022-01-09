 # completely ridiculous hack to import stuff properly. somebody save me from myself
import importlib
from allennlp.common.util import import_submodules
importlib.invalidate_caches()
import sys
sys.path.append(".")
import_submodules("qasrl")

import argparse
import json
import logging
import os
from typing import IO, List, Iterable, Tuple
import warnings
import struct

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

import numpy
import torch

from allennlp.common.file_utils import cached_path
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import lazy_groups_of, prepare_global_logging
from allennlp.common.checks import ConfigurationError
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.nn.util import remove_sentence_boundaries
from allennlp.modules.elmo import _ElmoBiLm, batch_to_ids
from allennlp.commands.subcommand import Subcommand

from qasrl.data.util import read_lines, get_verb_fields

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DEFAULT_OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json" # pylint: disable=line-too-long
DEFAULT_WEIGHT_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5" # pylint: disable=line-too-long
DEFAULT_BATCH_SIZE = 64

def empty_embedding() -> numpy.ndarray:
    return numpy.zeros((3, 0, 1024))

class ElmoEmbedder():
    def __init__(self,
                 options_file: str = DEFAULT_OPTIONS_FILE,
                 weight_file: str = DEFAULT_WEIGHT_FILE,
                 cuda_device: int = -1) -> None:
        """
        Parameters
        ----------
        options_file : ``str``, optional
            A path or URL to an ELMo options file.
        weight_file : ``str``, optional
            A path or URL to an ELMo weights file.
        cuda_device : ``int``, optional, (default=-1)
            The GPU device to run on.
        """
        self.indexer = ELMoTokenCharactersIndexer()

        logger.info("Initializing ELMo.")
        self.elmo_bilm = _ElmoBiLm(options_file, weight_file)
        if cuda_device >= 0:
            self.elmo_bilm = self.elmo_bilm.cuda(device=cuda_device)

        self.cuda_device = cuda_device

    def embed_batch(self, batch: List[List[str]], batch_metas) -> Tuple[List[Tuple[str, int]], List[torch.Tensor]]:
        """
        Parameters
        ----------
        batch : ``List[List[str]]``, required
            A list of tokenized sentences.
        batch_metas : ``List[Dict]``, required
            A list of metadata:
              sentence_id: str
              verb_indices: List[int]
        Returns
        -------
        """
        character_ids = batch_to_ids(batch)
        if self.cuda_device >= 0:
            character_ids = character_ids.cuda(device=self.cuda_device)

        bilm_output = self.elmo_bilm(character_ids)
        layer_activations_with_bos_eos = bilm_output['activations']
        mask_with_bos_eos = bilm_output['mask']

        top_activations = remove_sentence_boundaries(
            layer_activations_with_bos_eos[2], mask_with_bos_eos
        )[0]

        results = []
        for i, meta in enumerate(batch_metas):
            sid = meta["sentence_id"]
            for vi in meta["verb_indices"]:
                verb_id = {"sentenceId": sid, "verbIndex": vi}
                results.append((verb_id, top_activations[i, vi]))

        return results

    def get_qasrl_sentences(self, file_path: str):
        for line in read_lines(cached_path(file_path)):
            sentence_json = json.loads(line)
            verb_indices = [int(k) for k, _ in sentence_json["verbEntries"].items()]
            yield (sentence_json["sentenceTokens"], { "sentence_id": sentence_json["sentenceId"], "verb_indices": verb_indices })

    def get_propbank_sentences(self, file_path: str):
        raise NotImplementedError

    def embed_file(self,
                   input_path: str,
                   output_file_prefix: str,
                   is_propbank: bool = False,
                   batch_size: int = DEFAULT_BATCH_SIZE) -> None:

        def get_sentences():
            if not is_propbank:
                return self.get_qasrl_sentences(input_path)
            else:
                return self.get_propbank_sentences(input_path)

        with open(output_file_prefix + "_ids.jsonl", "w") as f_ids:
            with open(output_file_prefix + "_emb.bin", "wb") as f_emb:
                for sentence_batch in lazy_groups_of(Tqdm.tqdm(self.get_qasrl_sentences(input_path)), batch_size):
                    batch_sentences, batch_metas = map(list, zip(*sentence_batch))
                    for verb_id, emb in self.embed_batch(batch_sentences, batch_metas):
                        f_ids.write(json.dumps(verb_id) + "\n")
                        bs = emb.numpy().tobytes()
                        f_emb.write(bs)

def elmo_command(args):
    elmo_embedder = ElmoEmbedder(args.options_file, args.weight_file, args.cuda_device)
    # prepare_global_logging(os.path.realpath(os.path.dirname(args.output_file)), args.file_friendly_logging)

    with torch.no_grad():
        elmo_embedder.embed_file(
            args.input_path,
            args.output_file_prefix,
            args.is_propbank,
            args.batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Write ELMo vectors")
    parser.add_argument('--cuda_device', type=int, default=-1)
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_file_prefix', type=str)
    # subparser.add_argument('input_file', type=argparse.FileType('r', encoding='utf-8'),
    #                        help='The path to the input file.')
    parser.add_argument('--is_propbank', type=bool, default = False)
    parser.add_argument(
        '--options-file',
        type=str,
        default=DEFAULT_OPTIONS_FILE,
        help='The path to the ELMo options file.')
    parser.add_argument(
        '--weight-file',
        type=str,
        default=DEFAULT_WEIGHT_FILE,
        help='The path to the ELMo weight file.')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='The batch size to use.')
    parser.add_argument('--cuda-device', type=int, default=-1, help='The cuda_device to run on.')

    elmo_command(parser.parse_args())
