from typing import List, Dict

import torch

import math

from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn.modules import Linear, Dropout, Embedding, LSTMCell
import torch.nn.functional as F

from allennlp.common import Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import TimeDistributed

from qasrl.util.model_utils import block_orthonormal_initialization
from qasrl.data.util import get_slot_label_namespace

import logging
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class SlotSequenceGenerator(torch.nn.Module, Registrable):
    def __init__(self,
                 vocab: Vocabulary,
                 slot_names: List[str],
                 input_dim: int,
                 slot_hidden_dim: int = 100,
                 rnn_hidden_dim: int = 200,
                 slot_embedding_dim: int = 100,
                 num_layers: int = 1,
                 recurrent_dropout: float = 0.1,
                 highway: bool = True,
                 share_rnn_cell: bool =  False,
                 share_slot_hidden: bool = False,
                 clause_mode: bool = False): # clause_mode flag no longer used
        super(SlotSequenceGenerator, self).__init__()
        self.vocab = vocab
        self._slot_names = slot_names
        self._input_dim = input_dim
        self._slot_embedding_dim = slot_embedding_dim
        self._slot_hidden_dim = slot_hidden_dim
        self._rnn_hidden_dim = rnn_hidden_dim
        self._num_layers = num_layers
        self._recurrent_dropout = recurrent_dropout

        question_space_size = 1
        for slot_name in self._slot_names:
            num_values_for_slot = len(self.vocab.get_index_to_token_vocabulary(get_slot_label_namespace(slot_name)))
            question_space_size *= num_values_for_slot
            logger.info("%s values for slot %s" % (num_values_for_slot, slot_name))
        logger.info("Slot sequence generation space: %s possible sequences" % question_space_size)

        slot_embedders = []
        for i, n in enumerate(self.get_slot_names()[:-1]):
            num_labels = self.vocab.get_vocab_size(get_slot_label_namespace(n))
            assert num_labels > 0, "Slot named %s has 0 vocab size"%(n)
            embedder = Embedding(num_labels, self._slot_embedding_dim)
            self.add_module('embedder_%s'%n, embedder)
            slot_embedders.append(embedder)

        self._slot_embedders = slot_embedders

        self._highway = highway

        rnn_cells = []
        highway_nonlin = []
        highway_lin = []
        for l in range(self._num_layers):
            layer_cells = []
            layer_highway_nonlin = []
            layer_highway_lin = []
            shared_cell = None
            layer_input_size = self._input_dim + self._slot_embedding_dim if l == 0 else self._rnn_hidden_dim
            for i, n in enumerate(self._slot_names):
                if share_rnn_cell:
                    if shared_cell is None:
                        shared_cell = LSTMCell(layer_input_size, self._rnn_hidden_dim)
                        self.add_module('layer_%d_cell'%l, shared_cell)
                        if highway:
                            shared_highway_nonlin = Linear(layer_input_size + self._rnn_hidden_dim, self._rnn_hidden_dim)
                            shared_highway_lin = Linear(layer_input_size, self._rnn_hidden_dim, bias = False)
                            self.add_module('layer_%d_highway_nonlin'%l, shared_highway_nonlin)
                            self.add_module('layer_%d_highway_lin'%l, shared_highway_lin)
                    layer_cells.append(shared_cell)
                    if highway:
                        layer_highway_nonlin.append(shared_highway_nonlin)
                        layer_highway_lin.append(shared_highway_lin)
                else:
                    cell = LSTMCell(layer_input_size, self._rnn_hidden_dim)
                    cell.weight_ih.data.copy_(block_orthonormal_initialization(layer_input_size, self._rnn_hidden_dim, 4).t())
                    cell.weight_hh.data.copy_(block_orthonormal_initialization(self._rnn_hidden_dim, self._rnn_hidden_dim, 4).t())
                    self.add_module('layer_%d_cell_%s'%(l, n), cell)
                    layer_cells.append(cell)
                    if highway:
                        nonlin = Linear(layer_input_size + self._rnn_hidden_dim, self._rnn_hidden_dim)
                        lin = Linear(layer_input_size, self._rnn_hidden_dim, bias = False)
                        nonlin.weight.data.copy_(block_orthonormal_initialization(layer_input_size + self._rnn_hidden_dim, self._rnn_hidden_dim, 1).t())
                        lin.weight.data.copy_(block_orthonormal_initialization(layer_input_size, self._rnn_hidden_dim, 1).t())
                        self.add_module('layer_%d_highway_nonlin_%s'%(l, n), nonlin)
                        self.add_module('layer_%d_highway_lin_%s'%(l, n), lin)
                        layer_highway_nonlin.append(nonlin)
                        layer_highway_lin.append(lin)

            rnn_cells.append(layer_cells)
            highway_nonlin.append(layer_highway_nonlin)
            highway_lin.append(layer_highway_lin)

        self._rnn_cells = rnn_cells
        if highway:
            self._highway_nonlin = highway_nonlin
            self._highway_lin = highway_lin

        shared_slot_hidden = None
        slot_hiddens = []
        slot_preds = []
        slot_num_labels = []
        for i, n in enumerate(self._slot_names):
            num_labels = self.vocab.get_vocab_size(get_slot_label_namespace(n))
            slot_num_labels.append(num_labels)

            if share_slot_hidden:
                if shared_slot_hidden is None:
                    shared_slot_hidden = Linear(self._rnn_hidden_dim, self._slot_hidden_dim)
                    self.add_module('slot_hidden', shared_slot_hidden)
                slot_hiddens.append(shared_slot_hidden)
            else:
                slot_hidden = Linear(self._rnn_hidden_dim, self._slot_hidden_dim)
                slot_hiddens.append(slot_hidden)
                self.add_module('slot_hidden_%s'%n, slot_hidden)

            slot_pred = Linear(self._slot_hidden_dim, num_labels)
            slot_preds.append(slot_pred)
            self.add_module('slot_pred_%s'%n, slot_pred)

        self._slot_hiddens = slot_hiddens
        self._slot_preds = slot_preds
        self._slot_num_labels = slot_num_labels

        self._start_symbol = Parameter(torch.Tensor(self._slot_embedding_dim).normal_(0, 1))

    def get_slot_names(self):
        return self._slot_names

    def get_input_dim(self):
        return self._input_dim

    def _slot_quasi_recurrence(self,
                              slot_index,
                              slot_name,
                              inputs,
                              curr_embedding,
                              curr_mem):

        next_mem  = []
        curr_input = torch.cat([inputs, curr_embedding], -1)
        for l in range(self._num_layers):
            new_h, new_c = self._rnn_cells[l][slot_index](curr_input, curr_mem[l])
            if self._recurrent_dropout > 0:
                new_h = F.dropout(new_h, p = self._recurrent_dropout, training = self.training)
            next_mem.append((new_h, new_c))
            if self._highway:
                nonlin = self._highway_nonlin[l][slot_index](torch.cat([curr_input, new_h], -1))
                gate = torch.sigmoid(nonlin)
                curr_input = gate * new_h + (1. - gate) * self._highway_lin[l][slot_index](curr_input)
            else:
                curr_input = new_h
        hidden = F.relu(self._slot_hiddens[slot_index](new_h))
        logits = self._slot_preds[slot_index](hidden)

        return {
            "next_mem": next_mem,
            "logits": logits
        }

    def _init_recurrence(self, inputs):
        # start with batch_size start symbols and init multi-layer memory cell
        batch_size, _ = inputs.size()
        emb = self._start_symbol.view(1, -1).expand(batch_size, -1)
        mem = []
        for l in range(self._num_layers):
            mem.append((Variable(inputs.data.new().resize_(batch_size, self._rnn_hidden_dim).zero_()),
                        Variable(inputs.data.new().resize_(batch_size, self._rnn_hidden_dim).zero_())))
        return emb, mem

    def forward(self,
                inputs,
                **kwargs):
        batch_size, input_dim = inputs.size()

        slot_labels = {}
        for slot_name in self.get_slot_names():
            if slot_name in kwargs and kwargs[slot_name] is not None:
                slot_labels[slot_name] = kwargs[slot_name].squeeze(-1)
        if len(slot_labels) == 0:
            slot_labels = None

        # TODO check input_dim == input_dim

        curr_embedding, curr_mem = self._init_recurrence(inputs)
        slot_logits = {}
        for i, n in enumerate(self._slot_names):
            recurrence_dict = self._slot_quasi_recurrence(i, n, inputs, curr_embedding, curr_mem)
            slot_logits[n] = recurrence_dict["logits"]
            curr_mem = recurrence_dict["next_mem"]

            if i < len(self._slot_names) - 1:
                curr_embedding = self._slot_embedders[i](slot_labels[n])

        return slot_logits

    def beam_decode(self,
                    inputs, # shape: 1, input_dim
                    max_beam_size,
                    min_beam_probability,
                    clause_mode: bool = False):
        min_beam_log_probability = math.log(min_beam_probability)
        batch_size, input_dim = inputs.size()
        if batch_size != 1:
            raise ConfigurationError("beam_decode_single must be run with a batch size of 1.")
        if input_dim != self.get_input_dim():
            raise ConfigurationError("input dimension must match dimensionality of slot sequence model input.")

        ## metadata to recover sequences
        # slot_name -> List/Tensor of shape (beam_size) where value is index into slot's beam
        backpointers = {}
        # slot_name -> list (length <= beam_size) of indices indicating slot values
        slot_beam_labels = {}

        ## initialization for beam search loop
        init_embedding, init_mem = self._init_recurrence(inputs)
        # current state of the beam search: list of (input embedding, memory cells, log_prob), ordered by probability
        current_beam_states = [(init_embedding, init_mem, 0.)]

        for slot_index, slot_name in enumerate(self._slot_names):
            ending_clause_with_qarg = clause_mode and slot_index == (len(self._slot_names) - 1) and slot_name == "clause-qarg"
            # list of pairs (of backpointer, slot_value_index, new_embedding, new_mem, log_prob) ?
            candidate_new_beam_states = []
            for i, (emb, mem, prev_log_prob) in enumerate(current_beam_states):
                recurrence_dict = self._slot_quasi_recurrence(slot_index, slot_name, inputs, emb, mem)
                next_mem = recurrence_dict["next_mem"]
                logits = recurrence_dict["logits"].squeeze()
                log_probabilities = F.log_softmax(logits, -1)
                num_slot_values = self.vocab.get_vocab_size(get_slot_label_namespace(slot_name))
                for pred_slot_index in range(0, num_slot_values):
                    if len(log_probabilities.size()) == 0: # this only happens with slot vocab size of 1
                        log_prob = log_probabilities.item() + prev_log_prob
                    else:
                        log_prob = log_probabilities[pred_slot_index].item() + prev_log_prob
                    if slot_index < len(self._slot_names) - 1:
                        new_input_embedding = self._slot_embedders[slot_index](inputs.new([pred_slot_index]).long())
                    else:
                        new_input_embedding = None
                    # keep all expansions of the last step --- for now --- if we're on the qarg slot of a clause
                    if ending_clause_with_qarg or log_prob >= min_beam_log_probability:
                        candidate_new_beam_states.append((i, pred_slot_index, new_input_embedding, next_mem, log_prob))
            candidate_new_beam_states.sort(key = lambda t: t[4], reverse = True)
            # ditto the comment above; keeping all expansions of last step for clauses; we'll filter them later
            new_beam_states = candidate_new_beam_states[:max_beam_size] if not ending_clause_with_qarg else candidate_new_beam_states
            backpointers[slot_name] = [t[0] for t in new_beam_states]
            slot_beam_labels[slot_name] = [t[1] for t in new_beam_states]
            current_beam_states = [(t[2], t[3], t[4]) for t in new_beam_states]

        final_beam_size = len(current_beam_states)
        final_slots = {}
        for slot_name in reversed(self._slot_names):
            final_slots[slot_name] = inputs.new_zeros([final_beam_size], dtype = torch.int32)
        final_log_probs = inputs.new_zeros([final_beam_size], dtype = torch.float64)
        for beam_index in range(final_beam_size):
            final_log_probs[beam_index] = current_beam_states[beam_index][2]
            current_backpointer = beam_index
            for slot_name in reversed(self._slot_names):
                final_slots[slot_name][beam_index] = slot_beam_labels[slot_name][current_backpointer]
                current_backpointer = backpointers[slot_name][current_backpointer]

        # now if we're in clause mode, we need to filter the expanded beam
        if clause_mode:
            chosen_beam_indices = []
            for beam_index in range(final_beam_size):
                # TODO fix for abstracted slots, which have a different name. ...later. requires a nontrivial refactor
                qarg_name = self.vocab.get_token_from_index(final_slots["clause-qarg"][beam_index].item(), get_slot_label_namespace("clause-qarg"))
                qarg = "clause-%s" % qarg_name
                if qarg in self.get_slot_names():
                    # remove core arguments which are invalid
                    arg_value = self.vocab.get_token_from_index(final_slots[qarg][beam_index].item(), get_slot_label_namespace(qarg))
                    should_keep = arg_value != "_"
                else:
                    should_keep = True
                if should_keep:
                    chosen_beam_indices.append(beam_index)

            device = torch.device("cpu")
            if torch.cuda.is_available():
                device = torch.device("cuda:%s" % torch.cuda.current_device())
            chosen_beam_vector = torch.tensor(chosen_beam_indices, device = device).long()
            for slot_name in self._slot_names:
                final_slots[slot_name] = final_slots[slot_name].gather(0, chosen_beam_vector)
            final_log_probs = final_log_probs.gather(0, chosen_beam_vector)

        final_slot_indices = {
            slot_name: slot_indices.long().tolist()
            for slot_name, slot_indices in final_slots.items() }
        final_slot_labels = {
            slot_name: [self.vocab.get_token_from_index(index, get_slot_label_namespace(slot_name))
                        for index in slot_indices]
            for slot_name, slot_indices in final_slot_indices.items()
        }
        final_probs = final_log_probs.exp()
        return final_slot_indices, final_slot_labels, final_probs.tolist()
