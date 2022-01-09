import logging
from typing import Dict, List, Optional, Tuple, Set
import json

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import MetadataField
from allennlp.data import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, WordTokenizer

from qasrl.data.util import read_lines, get_verb_fields
from qasrl.data import QasrlFilter, QasrlInstanceReader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("qasrl")
class QasrlReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = { "tokens": SingleIdTokenIndexer(lowercase_tokens = True) },
                 qasrl_filter: QasrlFilter = QasrlFilter(),
                 instance_reader: QasrlInstanceReader = QasrlInstanceReader(),
                 include_metadata: bool = True,
                 lazy: bool = False):
        super().__init__(lazy)
        self._token_indexers = token_indexers
        self._qasrl_filter = qasrl_filter
        self._instance_reader = instance_reader
        self._include_metadata = include_metadata
        self._tokenizer = WordTokenizer()
        self._num_verbs = 0
        self._num_instances = 0

    @overrides
    def _read(self, file_list: str):
        self._num_verbs = 0
        self._num_instances = 0
        for file_path in file_list.split(","):
            if file_path.strip() == "":
                continue
            logger.info("Reading QASRL instances from dataset file at: %s", file_path)
            for line in read_lines(cached_path(file_path)):
                for instance in self.sentence_json_to_instances(json.loads(line)):
                    yield instance
        logger.info("Produced %d instances for %d verbs." % (self._num_instances, self._num_verbs))

    def sentence_json_to_instances(self, sentence_json, verbs_only = False):
        for verb_dict in self._qasrl_filter.filter_sentence(sentence_json):
            self._num_verbs += 1
            if verbs_only:
                instance_dict = get_verb_fields(self._token_indexers, verb_dict["sentence_tokens"], verb_dict["verb_index"])
                if self._include_metadata:
                    instance_dict["metadata"] = MetadataField(verb_dict)
                yield Instance(instance_dict)
            else:
                for instance_dict in self._instance_reader.read_instances(self._token_indexers, **verb_dict):
                    self._num_instances += 1
                    instance_metadata = instance_dict.pop("metadata", {})
                    if self._include_metadata:
                        instance_dict["metadata"] = MetadataField({**instance_metadata, **verb_dict})
                    yield Instance(instance_dict)

