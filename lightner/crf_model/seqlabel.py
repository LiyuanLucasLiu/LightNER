"""
.. module:: seqlabel
    :synopsis: sequence labeling model
 
.. moduleauthor:: Liyuan Liu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightner.utils as utils
from lightner.crf_model.crf import CRF
from lightner.crf_model.denselm import DenseLM

class SeqLabel(nn.Module):
    """
    Sequence Labeling model augumented with language model.

    Parameters
    ----------
    f_lm : ``torch.nn.Module``, required.
        The forward language modle for contextualized representations.
    b_lm : ``torch.nn.Module``, required.
        The backward language modle for contextualized representations.
    c_num : ``int`` , required.
        The number of characters.
    c_dim : ``int`` , required.
        The dimension of character embedding.
    c_hidden : ``int`` , required.
        The dimension of character hidden states.
    c_layer : ``int`` , required.
        The number of character lstms.
    w_num : ``int`` , required.
        The number of words.
    w_dim : ``int`` , required.
        The dimension of word embedding.
    w_hidden : ``int`` , required.
        The dimension of word hidden states.
    w_layer : ``int`` , required.
        The number of word lstms.
    y_num : ``int`` , required.
        The number of tags types.
    droprate : ``float`` , required
        The dropout ratio.
    unit : "str", optional, (default = 'lstm')
        The type of the recurrent unit.
    """
    def __init__(self, f_lm, b_lm, 
            c_num: int, 
            c_dim: int, 
            c_hidden: int, 
            c_layer: int, 
            w_num: int, 
            w_dim: int, 
            w_hidden: int, 
            w_layer: int, 
            y_num: int, 
            droprate: float, 
            unit: str = 'lstm'):
        super(SeqLabel, self).__init__()

        rnnunit_map = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}

        self.f_lm = f_lm
        self.b_lm = b_lm
        self.unit_type = unit

        self.char_embed = nn.Embedding(c_num, c_dim)
        self.word_embed = nn.Embedding(w_num, w_dim)

        self.char_seq = nn.Linear(c_hidden * 2, w_dim)
        if f_lm and b_lm:
            self.lm_seq = nn.Linear(f_lm.output_dim + b_lm.output_dim, w_dim)
            input_count = 3
        else:
            self.lm_seq = None
            input_count = 2

        self.relu = nn.ReLU()

        self.c_hidden = c_hidden
        tmp_rnn_dropout = droprate if c_layer > 1 else 0
        self.char_fw = rnnunit_map[unit](c_dim, c_hidden, c_layer, dropout = tmp_rnn_dropout)
        self.char_bw = rnnunit_map[unit](c_dim, c_hidden, c_layer, dropout = tmp_rnn_dropout)

        tmp_rnn_dropout = droprate if w_layer > 1 else 0
        self.word_rnn = rnnunit_map[unit](w_dim * input_count, w_hidden, w_layer, dropout = tmp_rnn_dropout, bidirectional = True)

        self.y_num = y_num

        self.crf = CRF(w_hidden * 2, y_num)

        self.drop = nn.Dropout(p = droprate)

    def to_params(self):
        """
        To parameters.
        """
        return {
            "model_type": "char-lstm-crf",
            "forward_lm": self.f_lm.to_params() if self.f_lm else None,
            "backward_lm": self.b_lm.to_params() if self.b_lm else None,
            "word_embed_num": self.word_embed.num_embeddings,
            "word_embed_dim": self.word_embed.embedding_dim,
            "char_embed_num": self.char_embed.num_embeddings,
            "char_embed_dim": self.char_embed.embedding_dim,
            "char_hidden": self.c_hidden,
            "char_layers": self.char_fw.num_layers,
            "word_hidden": self.word_rnn.hidden_size,
            "word_layers": self.word_rnn.num_layers,
            "droprate": self.drop.p,
            "y_num": self.y_num,
            "output_layer": "crf",
            "label_schema": "iobes",
            "unit_type": self.unit_type
        }

    @staticmethod
    def from_params(config):
        """
        From parameters.
        """
        assert(config['model_type'] == 'char-lstm-crf')
        if 'forward_lm' in config and config['forward_lm']:
            flm = DenseLM.from_params(config['forward_lm'])
        else:
            flm = None

        if 'backward_lm' in config and config['backward_lm']:
            blm = DenseLM.from_params(config['backward_lm'])
        else:
            blm = None

        return SeqLabel(flm, blm,
            config['char_embed_num'],
            config['char_embed_dim'],
            config['char_hidden'],
            config['char_layers'],
            config['word_embed_num'],
            config['word_embed_dim'],
            config['word_hidden'],
            config['word_layers'],
            config['y_num'],
            config['droprate'],
            config.get('unit_type', 'lstm'))

    def set_batch_seq_size(self, sentence):
        """
        Set the batch size and sequence length.
        """
        tmp = sentence.size()
        self.word_seq_length = tmp[0]
        self.batch_size = tmp[1]

    def load_pretrained_word_embedding(self, pre_word_embeddings):
        """
        Load pre-trained word embedding.
        """
        self.word_embed.weight = nn.Parameter(pre_word_embeddings)

    def forward(self, f_c, f_p, b_c, b_p, f_w, flm_w = None, blm_w = None, blm_ind = None):
        """
        Calculate the output (crf potentials).

        Parameters
        ----------
        f_c : ``torch.LongTensor``, required.
            Character-level inputs in the forward direction.
        f_p : ``torch.LongTensor``, required.
            Ouput position of character-level inputs in the forward direction.
        b_c : ``torch.LongTensor``, required.
            Character-level inputs in the backward direction.
        b_p : ``torch.LongTensor``, required.
            Ouput position of character-level inputs in the backward direction.
        f_w: ``torch.LongTensor``, required.
            Word-level inputs for the sequence labeling model.
        flm_w : ``torch.LongTensor``, required.
            Word-level inputs for the forward language model.
        blm_w : ``torch.LongTensor``, required.
            Word-level inputs for the backward language model.
        blm_ind : ``torch.LongTensor``, required.
            Ouput position of word-level inputs for the backward language model.
        

        Returns
        -------
        output: ``torch.FloatTensor``.
            A float tensor of shape (sequence_len, batch_size, from_tag_size, to_tag_size)
        """
        self.set_batch_seq_size(f_w)

        f_c_e = self.drop(self.char_embed(f_c))
        b_c_e = self.drop(self.char_embed(b_c))

        f_c_e, _ = self.char_fw(f_c_e)
        b_c_e, _ = self.char_bw(b_c_e)
        f_c_e = f_c_e.view(-1, self.c_hidden).index_select(0, f_p).view(self.word_seq_length, self.batch_size, self.c_hidden)
        b_c_e = b_c_e.view(-1, self.c_hidden).index_select(0, b_p).view(self.word_seq_length, self.batch_size, self.c_hidden)    
        c_o = self.drop(torch.cat([f_c_e, b_c_e], dim = 2))
        c_o = self.char_seq(c_o)

        w_e = self.word_embed(f_w)

        if flm_w is not None and self.f_lm and blm_w is not None and self.b_lm:
            self.f_lm.init_hidden()
            self.b_lm.init_hidden()
            f_lm_e = self.f_lm(flm_w)
            b_lm_e = self.b_lm(blm_w, blm_ind)
            lm_o = self.drop(torch.cat([f_lm_e, b_lm_e], dim = 2))
            lm_o = self.relu(self.lm_seq(lm_o))
            rnn_in = self.drop(torch.cat([c_o, lm_o, w_e], dim = 2))
        else:
            rnn_in = self.drop(torch.cat([c_o, w_e], dim = 2))

        rnn_out, _ = self.word_rnn(rnn_in)

        crf_out = self.crf(self.drop(rnn_out)).view(self.word_seq_length, self.batch_size, self.y_num, self.y_num)

        return crf_out
