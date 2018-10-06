"""
.. module:: sparse_lm
    :synopsis: sparse language model for sequence labeling
 
.. moduleauthor:: Liyuan Liu
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightner.utils as utils
from lightner.basic_modules.rnn import rnn_builder

class DenseLM(nn.Module):
    """
    The language model for the dense rnns with layer-wise selection.

    Parameters
    ----------
    ori_lm : ``torch.nn.Module``, required.
        the original module of language model.
    backward : ``bool``, required.
        whether the language model is backward.
    droprate : ``float``, required.
        the dropout ratrio.
    fix_rate: ``bool``, required.
        whether to fix the rqtio.
    """

    def __init__(self, rnn, backward, word_embed_num, word_embed_dim):
        super(DenseLM, self).__init__()

        self.rnn = rnn

        self.w_num = word_embed_num
        self.w_dim = word_embed_dim
        self.word_embed = nn.Embedding(word_embed_num, word_embed_dim)

        self.output_dim = rnn.output_dim

        self.backward = backward

    def to_params(self):
        """
        To parameters.
        """
        return {
            "backward": self.backward,
            "rnn_params": self.rnn.to_params(),
            "word_embed_num": self.word_embed.num_embeddings,
            "word_embed_dim": self.word_embed.embedding_dim
        }

    @staticmethod
    def from_params(config):
        """
        From parameters.
        """
        rnn = rnn_builder(config['rnn_params'])
        return DenseLM(rnn,
            config['backward'],
            config['word_embed_num'],
            config['word_embed_dim'])

    def prune_dense_rnn(self):
        """
        Prune dense rnn to be smaller by delecting layers.
        """
        prune_mask = self.rnn.prune_dense_rnn()
        self.output_dim = self.rnn.output_dim
        return prune_mask

    def init_hidden(self):
        """
        initialize hidden states.
        """
        self.rnn.init_hidden()

    def regularizer(self):
        """
        Calculate the regularization term.

        Returns
        ----------
        reg: ``list``.
            The list of regularization terms.
        """
        return self.rnn.regularizer()

    def prox(self):
        """
        the proximal calculator.
        """
        return self.rnn.prox()

    def forward(self, w_in, ind=None):
        """
        Calculate the output.

        Parameters
        ----------
        w_in : ``torch.LongTensor``, required.
            the input tensor, of shape (seq_len, batch_size).
        ind : ``torch.LongTensor``, optional, (default=None).
            the index tensor for the backward language model, of shape (seq_len, batch_size).

        Returns
        ----------
        output: ``torch.FloatTensor``.
            The ELMo outputs.
        """
        w_emb = self.word_embed(w_in)
        
        out = self.rnn(w_emb)

        if self.backward:
            out_size = out.size()
            out = out.view(out_size[0] * out_size[1], out_size[2]).index_select(0, ind).contiguous().view(out_size)

        return out