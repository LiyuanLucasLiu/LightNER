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

class tl_predict(object):
    """Base class for prediction, provide method to calculate f1 score and accuracy.

    Parameters
    ----------
    device: ``torch.device``, required.
        The target device for the dataset loader.
    tl_map: ``dict``, required.
        The dict for label to number.
    label_seq: ``str``, optional, (default = "string").
        Whether to decode label sequence or inline marks.
    batch_size: ``int``, optional, (default = 50).
        The batch size for decoding.
    """
    def __init__(self, device, tl_map, label_seq = "string", batch_size = 50):
        self.device = device
        self.tl_map = tl_map
        self.r_tl_map = {v: k for k, v in tl_map.items()}
        self.batch_size = batch_size
        self.decode_str = {"string": self.decode_s, "label": self.decode_l}[label_seq]

    def decode_l(self, features, tc_label, tl_label):
        """
        decode a sentence coupled with label

        Parameters
        ----------
        features: ``list``, required.
            List of words list
        tc_label: ``list``, required.
            Chunk label list.
        tl_label: ``list``, required.
            Type label list.
        """
        chunks = list()
        chunk_ind = 0
        type_ind = 0

        for feature in features:
            tmp_chunks = ""

            previous_type = "None"

            for word in feature:

                if tc_label[chunk_ind] > 0:
                    if previous_type != "None":
                        tmp_chunks += '\t' + previous_type + '\n'
                    previous_type = self.r_tl_map[tl_label[type_ind].item()]
                    type_ind += 1
                    if previous_type == "None":
                        tmp_chunks += word + '\n'
                    else:
                        tmp_chunks += word
                else:
                    if previous_type != "None":
                        tmp_chunks += ' ' + word
                    else:
                        tmp_chunks += word + '\n'

                chunk_ind += 1 

            if previous_type != "None":
                tmp_chunks += '\t' + previous_type + '\n'
            chunk_ind += 1
            type_ind += 1

            chunks.append(tmp_chunks)

        return chunks

    def decode_s(self, features, tc_label, tl_label):
        """
        decode a sentence in the format of <>

        Parameters
        ----------
        features: ``list``, required.
            List of words list
        tc_label: ``list``, required.
            Chunk label list.
        tl_label: ``list``, required.
            Type label list.
        """
        chunks = list()
        chunk_ind = 0
        type_ind = 0

        for feature in features:
            tmp_chunks = ""
            previous_type = "None"

            for word in feature:

                if tc_label[chunk_ind] > 0:
                    if previous_type != "None":
                        tmp_chunks += "</{}> ".format(previous_type)
                    previous_type = self.r_tl_map[tl_label[type_ind].item()]
                    type_ind += 1
                    if previous_type != "None":
                        tmp_chunks += "<{}> ".format(previous_type)
                    tmp_chunks += word + ' '
                else:
                    tmp_chunks += word + ' '

                chunk_ind += 1 

            if previous_type != "None":
                tmp_chunks += "</{}>".format(previous_type)

            chunk_ind += 1
            type_ind += 1

            tmp_chunks = tmp_chunks.rstrip()

            chunks.append(tmp_chunks)

        assert (chunk_ind == len(tc_label))
        assert (type_ind == len(tl_label) + 1)

        return chunks

    def output_batch(self, ner_model, documents):
        """
        decode the whole corpus in the specific format by calling apply_model to fit specific models

        Parameters
        ----------
        ner_model: ``nn.Module``, required.
            Sequence labeling model.
        feature: ``list``, required.
            List of list of list of str (list of documents, which is list of sentences, which is list of str).
            Or list of list of str.
            Or list of str.
        """
        if not documents or 0 == len(documents):
            return list()

        ner_model.eval()

        if type(documents[0]) != list:

            chunk_label, type_label = self.apply_model(ner_model, [documents])
            output_file = self.decode_str([documents], chunk_label, type_label)[0]

        elif type(documents[0][0]) != list:

            output_file = list()
            features = documents
            f_len = len(features)
            for ind in tqdm(range(0, f_len, self.batch_size)):
                eind = min(f_len, ind + self.batch_size)
                chunk_label, type_label = self.apply_model(ner_model, features[ind: eind])
                output_file.extend(self.decode_str(features[ind: eind], chunk_label, type_label))

        elif type(documents[0][0][0]) != list:

            tmp_output_file = list()
            document_len = [0] + [len(doc) for doc in documents]
            flat_sent = [sent for doc in documents for sent in doc]
            f_len = len(flat_sent)
            assert (sum(document_len) == f_len)
            document_len = list(itertools.accumulate(document_len))

            for ind in tqdm(range(0, f_len, self.batch_size)):
                eind = min(f_len, ind + self.batch_size)
                chunk_label, type_label = self.apply_model(ner_model, flat_sent[ind:eind])
                tmp_output_file.extend(self.decode_str(flat_sent[ind:eind], chunk_label, type_label))

            output_file = [tmp_output_file[document_len[ind]: document_len[ind+1]] \
                                for ind in range(len(document_len) - 1)]

        else:
            raise Exception("Wrong Format! Only list of str, list of list of str or list of list of list of str are accepted.")

        return output_file
        
    def apply_model(self, ner_model, features):
        """
        template function for apply_model

        args:
            ner_model: sequence labeling model
            feature (list): list of words list
        """
        return None

class predict_wc_tl(tl_predict):
    """prediction class for AutoNER

    args: 
        if_cuda: if use cuda to speed up 
        f_map: dictionary for words
        c_map: dictionary for chars
        tl_map: dictionary for labels
        pad_word: word padding
        pad_char: word padding
        pad_label: label padding
        start_label: start label 
        label_seq: type of decode function, set `True` to couple label with text, or set 'False' to insert label into test
        batch_size: size of batch in decoding
        caseless: caseless or not
    """
   
    def __init__(self, device,
            w_map: dict, 
            c_map: dict, 
            tl_map: dict,
            label_seq: bool = True,
            batch_size: int = 50):
        tl_predict.__init__(self, device, tl_map, label_seq, batch_size)
        self.w_map = w_map
        self.c_map = c_map
        self.tl_map = tl_map

        self.w_st, self.w_unk, self.w_con, self.w_pad = w_map['<s>'], w_map['<unk>'], w_map['< >'], w_map['<\n>']
        self.c_st, self.c_unk, self.c_con, self.c_pad = c_map['<s>'], c_map['<unk>'], c_map['< >'], c_map['<\n>']

    def apply_model(self, ner_model, features):
        """
        apply_model function for LM-LSTM-CRF

        args:
            ner_model: sequence labeling model
            feature (list): list of words list
        """
        cur_batch_size = len(features)

        tmp_b0, tmp_b1, tmp_b2 = list(), list(), list()
        for f_l in features:
            tmp_w = [self.w_st, self.w_con]
            tmp_c = [self.c_st, self.c_con]
            tmp_mc = [0, 1]

            for i_f in f_l:
                tmp_w = tmp_w + [self.w_map.get(i_f, self.w_map.get(i_f.lower(), self.w_unk))] * len(i_f) + [self.w_con]
                tmp_c = tmp_c + [self.c_map.get(t, self.c_unk) for t in i_f] + [self.c_con]
                tmp_mc = tmp_mc + [0] * len(i_f) + [1]

            tmp_w.append(self.w_pad)
            tmp_c.append(self.c_pad)
            tmp_mc.append(0)

            tmp_b0.append(tmp_w)
            tmp_b1.append(tmp_c)
            tmp_b2.append(tmp_mc)

        csl = max([len(tup) for tup in tmp_b0])

        word_t = torch.LongTensor([tup + [self.w_pad] * (csl - len(tup)) for tup in tmp_b0]).to(self.device)
        char_t = torch.LongTensor([tup + [self.c_pad] * (csl - len(tup)) for tup in tmp_b1]).to(self.device)
        chunk_m = torch.ByteTensor([tup + [0] * (csl - len(tup)) for tup in tmp_b2]).to(self.device)

        output = ner_model(word_t, char_t, chunk_m)

        chunk_score = ner_model.chunking(output)
        pred_chunk = (chunk_score < 0)
        type_score = ner_model.typing(output, pred_chunk)
        pred_type = type_score.argmax(dim = 1)

        pred_chunk = pred_chunk.cpu()
        pred_type = pred_type.data.cpu()

        return pred_chunk, pred_type
