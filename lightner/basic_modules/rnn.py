"""
.. module:: rnn
    :synopsis: rnn
 
.. moduleauthor:: Liyuan Liu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from lightner.basic_modules.ld_rnn import SDRNN
from lightner.basic_modules.basic_rnn import BasicRNN

def rnn_builder(config):
    """
    To parameters.
    """
    rnn_type_dict = {'Basic': BasicRNN, 'LDRNN': SDRNN}
    return rnn_type_dict[config['rnn_type']].from_params(config)
