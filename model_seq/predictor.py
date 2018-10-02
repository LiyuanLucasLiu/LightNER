"""
.. module:: predictor
    :synopsis: prediction method (for un-annotated text)
 
.. moduleauthor:: Liyuan Liu
"""

import torch
import torch.autograd as autograd
import numpy as np
import itertools
import sys
from tqdm import tqdm

from model.crf import CRFDecode

class predict(object):
    """Base class for prediction, provide method to calculate f1 score and accuracy 

    Parameters
    ----------
    device: ``torch.device``, required.
        The target device for the dataset loader.
    l_map: ``dict``, required.
        The dict for label to number.
    label_seq: ``bool``, optional, (default = True).
        Whether to decode label sequence or inline marks.
    batch_size: ``int``, optional, (default = 50).
        The batch size for decoding.
    """

    def __init__(self, device, l_map, label_seq = True, batch_size = 50):
        self.device = device
        self.l_map = l_map
        self.r_l_map = {v: k for k, v in l_map.items()}
        self.batch_size = batch_size
        if label_seq:
            self.decode_str = self.decode_l
        else:
            self.decode_str = self.decode_s

    def decode_l(self, feature, label):
        """
        decode a sentence coupled with label

        Parameters
        ----------
        feature: ``list``, required.
            Words list
        label: ``list``, required.
            Label list.
        """
        return '\n'.join(map(lambda t: t[0] + ' '+ self.r_l_map[t[1]], zip(feature, label)))

    def decode_s(self, feature, label):
        """
        decode a sentence in the format of <>

        Parameters
        ----------
        feature: ``list``, required.
            Words list
        label: ``list``, required.
            Label list.
        """
        chunks = ""
        current = None

        for f, y in zip(feature, label):
            label = self.r_l_map[y]

            if label.startswith('B-'):

                if current is not None:
                    chunks += "</"+current+"> "
                current = label[2:]
                chunks += "<"+current+"> " + f + " "

            elif label.startswith('S-'):

                if current is not None:
                    chunks += " </"+current+"> "
                current = label[2:]
                chunks += "<"+current+"> " + f + " </"+current+"> "
                current = None

            elif label.startswith('I-'):

                if current is not None:
                    base = label[2:]
                    if base == current:
                        chunks += f+" "
                    else:
                        chunks += "</"+current+"> <"+base+"> " + f + " "
                        current = base
                else:
                    current = label[2:]
                    chunks += "<"+current+"> " + f + " "

            elif label.startswith('E-'):

                if current is not None:
                    base = label[2:]
                    if base == current:
                        chunks += f + " </"+base+"> "
                        current = None
                    else:
                        chunks += "</"+current+"> <"+base+"> " + f + " </"+base+"> "
                        current = None

                else:
                    current = label[2:]
                    chunks += "<"+current+"> " + f + " </"+current+"> "
                    current = None

            else:
                if current is not None:
                    chunks += "</"+current+"> "
                chunks += f+" "
                current = None

        if current is not None:
            chunks += "</"+current+"> "

        return chunks

    def output_batch(self, ner_model, documents):
        """
        decode the whole corpus in the specific format by calling apply_model to fit specific models

        Parameters
        ----------
        ner_model: ``nn.Module``, required.
            Sequence labeling model.
        feature: ``list``, required.
            List of list of words (list of documents, which is list of sentences, which is list of words).
        """
        ner_model.eval()
        d_len = len(documents)
        output_file = ""

        for d_ind in tqdm( range(0, d_len), mininterval=1,
                desc=' - Process', leave=False, file=sys.stdout):
            output_file = output_file + '-DOCSTART- -DOCSTART- -DOCSTART-\n\n'
            features = documents[d_ind]
            f_len = len(features)
            for ind in range(0, f_len, self.batch_size):
                eind = min(f_len, ind + self.batch_size)
                labels = self.apply_model(ner_model, features[ind: eind])
                labels = torch.unbind(labels, 1)

                for ind2 in range(ind, eind):
                    f = features[ind2]
                    l = labels[ind2 - ind][0: len(f) ]
                    output_file = output_file + self.decode_str(features[ind2], l) + '\n\n'

        return output_file
        
    def apply_model(self, ner_model, features):
        """
        template function for apply_model

        args:
            ner_model: sequence labeling model
            feature (list): list of words list
        """
        return None

class predict_wc(predict):
    """prediction class for LM-LSTM-CRF

    args: 
        if_cuda: if use cuda to speed up 
        f_map: dictionary for words
        c_map: dictionary for chars
        l_map: dictionary for labels
        pad_word: word padding
        pad_char: word padding
        pad_label: label padding
        start_label: start label 
        label_seq: type of decode function, set `True` to couple label with text, or set 'False' to insert label into test
        batch_size: size of batch in decoding
        caseless: caseless or not
    """
   
    def __init__(self, device,
            flm_map: dict, 
            blm_map: dict, 
            gw_map: dict, 
            c_map: dict, 
            y_map: dict,
            label_seq: bool = True,
            batch_size: int = 50):
        predict.__init__(self, device, l_map, label_seq, batch_size)
        self.decoder = CRFDecode(l_map)
        self.flm_map = flm_map
        self.blm_map = blm_map
        self.gw_map = gw_map
        self.c_map = c_map
        self.y_map = y_map

        self.c_unk = c_map['<unk>']
        self.flm_unk = flm_map['<unk>']
        self.blm_unk = blm_map['<unk>']
        self.gw_unk = gw_map['<unk>']

        self.c_pad = c_map['\n']
        self.gw_pad = gw_map['<\n>']
        self.flm_pad = flm_map['\n']
        self.blm_pad = blm_map['\n']

    def apply_model(self, ner_model, features):
        """
        apply_model function for LM-LSTM-CRF

        args:
            ner_model: sequence labeling model
            feature (list): list of words list
        """
        cur_batch_size = len(features)

        char_len = [[len(tup) + 1 for tup in sentence] for sentence in features]
        char_inses = [tup for sentence in features for tup in (sentence + ' ')]

        char_padded_len = max([len(tup) for tup in char_inses])
        word_padded_len = max([len(tup) for tup in features])

        tmp_batch =  [list() for ind in range(11)]

        for instance_ind in range(cur_batch_size):

            char_f = [self.c_map.get(tup, self.c_unk) for tup in char_inses[instance_ind]]
            gw_f = [self.w_map.get(tup, self.gw_unk) for tup in features[instance_ind]]
            flm_f = [self.flm_map.get(tup, self.flm_unk) for tup in features[instance_ind]]
            blm_f = [self.blm_map.get(tup, self.blm_unk) for tup in features[instance_ind]]

            char_padded_len_ins = char_padded_len - len(char_f)
            word_padded_len_ins = word_padded_len - len(word_f)

            tmp_batch[0].append(char_f + [self.c_pad] + [self.c_pad] * char_padded_len_ins)
            tmp_batch[2].append([self.c_pad] + char_f[::-1] + [self.c_pad] * char_padded_len_ins)

            tmp_p = list( itertools.accumulate(char_len+[1]+[0]* word_padded_len_ins) )
            tmp_batch[1].append([(x - 1) * cur_batch_size + instance_ind for x in tmp_p])
            tmp_p = list(itertools.accumulate([1]+char_len[::-1]))[::-1] + [1]*word_padded_len_ins
            tmp_batch[3].append([(x - 1) * cur_batch_size + instance_ind for x in tmp_p])

            tmp_batch[4].append(gw_f + [self.gw_pad] + [self.gw_pad] * word_padded_len_ins)

            tmp_batch[5].append(flm_f + [self.flm_pad] + [self.flm_pad] * word_padded_len_ins)
            tmp_batch[6].append([self.blm_pad] + blm_f[::-1] + [self.blm_pad] * word_padded_len_ins)

            tmp_p = list(range(len(blm_f), -1, -1)) + list(range(len(blm_f)+1, word_padded_len+1))
            tmp_batch[7].append([x * cur_batch_size + instance_ind for x in tmp_p])

            tmp_batch[8].append([1] * len(gw_f) + [1] + [0] * word_padded_len_ins)

        tbt = [torch.LongTensor(v).transpose(0, 1).contiguous() for v in tmp_batch[0:8]] + [torch.ByteTensor(tmp_batch[8]).transpose(0, 1).contiguous()]

        tbt[1] = tbt[1].view(-1)
        tbt[3] = tbt[3].view(-1)
        tbt[7] = tbt[7].view(-1)

        f_c, f_p, b_c, b_p, f_w, flm_w, blm_w, blm_ind, mask_v = [ten.to(device) for ten in tbt]

        scores = seq_model(f_c, f_p, b_c, b_p, f_w, flm_w, blm_w, blm_ind)
        
        decoded = self.decoder.decode(scores.data, mask_v)

        return decoded
