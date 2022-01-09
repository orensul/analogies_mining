from typing import List, Dict

import torch

from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn.modules import Linear, Dropout, Embedding, LSTMCell
import torch.nn.functional as F

from allennlp.common import Params, Registrable
from allennlp.data import Vocabulary
from allennlp.modules import TimeDistributed

from qasrl.util.model_utils import block_orthonormal_initialization
from qasrl.data.util import get_slot_label_namespace

class SlotSequenceEncoder(torch.nn.Module, Registrable):
    def __init__(self,
            vocab: Vocabulary,
            slot_names: List[str],
            input_dim: int,
            slot_embedding_dim: int = 100,
            output_dim: int = 200,
            num_layers: int = 1,
            recurrent_dropout: float = 0.1,
            highway: bool = True,
            share_rnn_cell: bool =  False):
        super(SlotSequenceEncoder, self).__init__()
        self._vocab = vocab
        self._slot_names = slot_names
        self._input_dim = input_dim
        self._slot_embedding_dim = slot_embedding_dim
        self._output_dim = output_dim
        self._num_layers = num_layers
        self._recurrent_dropout = recurrent_dropout
        self._highway = highway
        self._share_rnn_cell = share_rnn_cell

        slot_embedders = []
        for i, n in enumerate(self.get_slot_names()):
            num_labels = self._vocab.get_vocab_size(get_slot_label_namespace(n))
            assert num_labels > 0, "Slot named %s has 0 vocab size"%(n)
            embedder = Embedding(num_labels, self._slot_embedding_dim)
            self.add_module('embedder_%s'%n, embedder)
            slot_embedders.append(embedder)
        self._slot_embedders = slot_embedders

        rnn_cells = []
        highway_nonlin = []
        highway_lin = []
        for l in range(self._num_layers):
            layer_cells = []
            layer_highway_nonlin = []
            layer_highway_lin = []
            shared_cell = None
            layer_input_size = self.get_input_dim() + self._slot_embedding_dim if l == 0 else self._output_dim
            for i, n in enumerate(self._slot_names):
                if share_rnn_cell:
                    if shared_cell is None:
                        shared_cell = LSTMCell(layer_input_size, self._output_dim)
                        self.add_module('layer_%d_cell'%l, shared_cell)
                        if highway:
                            shared_highway_nonlin = Linear(layer_input_size + self._output_dim, self._output_dim)
                            shared_highway_lin = Linear(layer_input_size, self._output_dim, bias = False)
                            self.add_module('layer_%d_highway_nonlin'%l, shared_highway_nonlin)
                            self.add_module('layer_%d_highway_lin'%l, shared_highway_lin)
                    layer_cells.append(shared_cell)
                    if highway:
                        layer_highway_nonlin.append(shared_highway_nonlin)
                        layer_highway_lin.append(shared_highway_lin)
                else:
                    cell = LSTMCell(layer_input_size, self._output_dim)
                    cell.weight_ih.data.copy_(block_orthonormal_initialization(layer_input_size, self._output_dim, 4).t())
                    cell.weight_hh.data.copy_(block_orthonormal_initialization(self._output_dim, self._output_dim, 4).t())
                    self.add_module('layer_%d_cell_%s'%(l, n), cell)
                    layer_cells.append(cell)
                    if highway:
                        nonlin = Linear(layer_input_size + self._output_dim, self._output_dim)
                        lin = Linear(layer_input_size, self._output_dim, bias = False)
                        nonlin.weight.data.copy_(block_orthonormal_initialization(layer_input_size + self._output_dim, self._output_dim, 1).t())
                        lin.weight.data.copy_(block_orthonormal_initialization(layer_input_size, self._output_dim, 1).t())
                        self.add_module('layer_%d_highway_nonlin_%s'%(l, n), nonlin)
                        self.add_module('layer_%d_highway_lin_%s'%(l, n), lin)
                        layer_highway_nonlin.append(nonlin)
                        layer_highway_lin.append(lin)

            rnn_cells.append(layer_cells)
            highway_nonlin.append(layer_highway_nonlin)
            highway_lin.append(layer_highway_lin)

        self._rnn_cells = rnn_cells
        if self._highway:
            self._highway_nonlin = highway_nonlin
            self._highway_lin = highway_lin

    def forward(self,
                pred_reps,
                slot_labels: Dict[str, torch.LongTensor]):
        # Shape: batch_size, numpred_rep_dim
        batch_size, pred_rep_dim = pred_reps.size()

        # initialize the memory cells
        curr_mem = []
        for l in range(self._num_layers):
            curr_mem.append((Variable(pred_reps.data.new().resize_(batch_size, self._output_dim).zero_()),
                             Variable(pred_reps.data.new().resize_(batch_size, self._output_dim).zero_())))

        last_h = None
        for i, n in enumerate(self._slot_names):
            next_mem  = []
            curr_embedding = self._slot_embedders[i](slot_labels[n])
            curr_input = torch.cat([pred_reps, curr_embedding], -1)
            for l in range(self._num_layers):
                new_h, new_c = self._rnn_cells[l][i](curr_input, curr_mem[l])
                if self._recurrent_dropout > 0:
                    new_h = F.dropout(new_h, p = self._recurrent_dropout, training = self.training)
                next_mem.append((new_h, new_c))
                if self._highway:
                    nonlin = self._highway_nonlin[l][i](torch.cat([curr_input, new_h], -1))
                    gate = torch.sigmoid(nonlin)
                    curr_input = gate * new_h + (1. - gate) * self._highway_lin[l][i](curr_input)
                else:
                    curr_input = new_h
            curr_mem = next_mem
            last_h = new_h

        return last_h

    def get_slot_names(self):
        return self._slot_names

    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim
