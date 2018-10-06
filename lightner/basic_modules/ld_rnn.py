"""
.. module:: ld_rnn
    :synopsis: densely connected rnn
 
.. moduleauthor:: Liyuan Liu
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightner.utils as utils

from lightner.basic_modules.basic_rnn import RNN

class SBUnit(nn.Module):
    """
    The basic recurrent unit for the dense-RNNs wrapper.

    Parameters
    ----------
    unit : ``torch.nn.Module``, required.
        The type of rnn unit.
    input_dim : ``float``, required.
        The input dimension fo the unit.
    increase_rate : ``float``, required.
        The hidden dimension fo the unit.
    droprate : ``float``, required.
        The dropout ratrio.
    """
    def __init__(self, unit, input_dim, increase_rate, droprate):
        super(SBUnit, self).__init__()

        rnnunit_map = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}

        self.unit_type = unit

        self.layer = rnnunit_map[unit](input_dim, increase_rate, 1)

        self.droprate = droprate

        self.input_dim = input_dim
        self.increase_rate = increase_rate
        self.output_dim = input_dim + increase_rate

        self.init_hidden()

    def init_hidden(self):
        """
        Initialize hidden states.
        """
        self.hidden_state = None

    def forward(self, x, weight=1):
        """
        Calculate the output.

        Parameters
        ----------
        x : ``torch.FloatTensor``, required.
            The input tensor, of shape (seq_len, batch_size, input_dim).
        weight : ``torch.FloatTensor``, required.
            The selection variable.

        Returns
        ----------
        output: ``torch.FloatTensor``.
            The output of RNNs.
        """
        if self.droprate > 0:
            new_x = F.dropout(x, p=self.droprate, training=self.training)
        else:
            new_x = x

        out, new_hidden = self.layer(new_x, self.hidden_state)

        self.hidden_state = utils.repackage_hidden(new_hidden)

        out = weight * out

        out = out.contiguous()

        return torch.cat([x, out], 2)

class SDRNN(RNN):
    """
    The multi-layer recurrent networks for the dense-RNNs wrapper.

    Parameters
    ----------
    ori_unit : ``torch.nn.Module``, required.
        the original module of rnn unit.
    droprate : ``float``, required.
        the dropout ratrio.
    fix_rate: ``bool``, required.
        whether to fix the rqtio.
    """
    def __init__(self, layer_num, unit, emb_dim, hid_dim, droprate, after_pruned = False):
        super(SDRNN, self).__init__()

        self.unit_type = unit
        self.layer_list = [SBUnit(unit, emb_dim + i * hid_dim, hid_dim, droprate) for i in range(layer_num)]
        self.output_dim = self.layer_list[-1].output_dim if layer_num > 0 else emb_dim

        self.layer = nn.Sequential(*self.layer_list)
        self.weight_list = nn.Parameter(torch.FloatTensor([1.0] * len(self.layer_list))) if (layer_num > 0 and after_pruned) else None

        self.init_hidden()

    def init_hidden(self):
        """
        Initialize hidden states.
        """
        for tup in self.layer_list:
            tup.init_hidden()

    def to_params(self):
        """
        To parameters.
        """
        return {
            "rnn_type": "LDRNN",
            "unit_type": self.unit_type,
            "layer_num": 0 if not self.layer else len(self.layer),
            "emb_dim": -1 if not self.layer else self.layer[0].input_dim,
            "hid_dim": -1 if not self.layer else self.layer[0].increase_rate,
            "droprate": -1 if not self.layer else self.layer[0].droprate,
            "after_pruned": True
        }

    @staticmethod
    def from_params(config):
        """
        From parameters.
        """
        assert (config['rnn_type'] == 'LDRNN')

        return SDRNN(config['layer_num'],
            config['unit_type'],
            config['emb_dim'],
            config['hid_dim'],
            config['droprate'],
            config['after_pruned'])

    def prox(self):
        """
        the proximal calculator.
        """
        self.weight_list.data.masked_fill_(self.weight_list.data < 0, 0)
        self.weight_list.data.masked_fill_(self.weight_list.data > 1, 1)
        none_zero_count = (self.weight_list.data > 0).sum()
        return none_zero_count

    def regularizer(self):
        """
        Calculate the regularization term.

        Returns
        ----------
        reg0: ``torch.FloatTensor``.
            The value of reg0.
        reg1: ``torch.FloatTensor``.
            The value of reg1.
        reg2: ``torch.FloatTensor``.
            The value of reg2.
        """
        reg3 = (self.weight_list * (1 - self.weight_list)).sum()
        none_zero = self.weight_list.data > 0
        none_zero_count = none_zero.sum()
        reg0 = none_zero_count
        reg1 = self.weight_list[none_zero].sum()
        return reg0, reg1, reg3

    def forward(self, x):
        """
        Calculate the output.

        Parameters
        ----------
        x : ``torch.FloatTensor``, required.
            the input tensor, of shape (seq_len, batch_size, input_dim).

        Returns
        ----------
        output: ``torch.FloatTensor``.
            The ELMo outputs.
        """
        if self.layer_list is not None:
            for ind in range(len(self.layer_list)):
                x = self.layer[ind](x, self.weight_list[ind])
        return x
