from torch_scope import wrapper

from lightner.crf_model.seqlabel import SeqLabel 
from lightner.crf_model.predictor import predict_wc

from lightner.two_level_model.predictor import predict_wc_tl
from lightner.two_level_model.autoner import AutoNER

from lightner.utils import read_conll_features 

import torch
import codecs
import logging
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 50

class decoder(object):
    """
    Abstract class for decoder.

    Parameters
    ----------
    model_file: ``dict``, required.
        Loaded checkpoint.
    pw: ``wrapper``, required.
        torch_scope wrapper for logging.
    configs: ``dict``, optional, (default = "{}").
        Additional configs.
    """
    def __init__(self,
                model_file: dict,
                pw: wrapper,
                configs: dict = {}):

        raise NotImplementedError

    def decode(self, documents):
        """
        Decode documents.

        Parameters
        ----------
        documents: ``list``, required.
            List of str or list of list of str.
        """
        raise NotImplementedError


class decoder_tl(decoder):
    """
    Decode function for AutoNER models.

    Parameters
    ----------
    model_file: ``dict``, required.
        Loaded checkpoint.
    pw: ``wrapper``, required.
        torch_scope wrapper for logging.
    configs: ``dict``, optional, (default = "{}").
        Additional configs.
    """
    def __init__(self,
                model_file: dict,
                pw: wrapper,
                configs: dict = {}):

        self.pw = pw
        gpu_index = self.pw.auto_device() if 'auto' == configs.get('gpu', 'auto') else int(configs.get('gpu', 'auto'))
        self.device = torch.device("cuda:" + str(gpu_index) if gpu_index >= 0 else "cpu")
        if gpu_index >= 0:
            torch.cuda.set_device(gpu_index)

        name_list = ['w_map', 'c_map', 'tl_map', 'config', 'model']
        w_map, c_map, tl_map, config, model_param = [model_file[tup] for tup in name_list]

        logger.info('Building sequence labeling model.')
        self.ner_model = AutoNER.from_params(config)
        self.ner_model.load_state_dict(model_param)
        self.ner_model.to(self.device)
        self.ner_model.eval()

        logger.info('Building predictor.')
        self.predictor = predict_wc_tl(self.device, w_map, c_map, tl_map, \
                        label_seq = configs.get("label_seq", "string"), \
                        batch_size = configs.get("batch_size", DEFAULT_BATCH_SIZE))

        logger.info('Model is ready.')

    def decode(self, documents):
        """
        Decode documents.

        Parameters
        ----------
        documents: ``list``, required.
            List of str or list of list of str or str.
        """
        self.ner_model.eval()
        return self.predictor.output_batch(self.ner_model, documents)


class decoder_wc(decoder):
    """
    Decode function for char-lstm-crf model.

    Parameters
    ----------
    model_file: ``dict``, required.
        Loaded checkpoint.
    pw: ``wrapper``, required.
        torch_scope wrapper for logging.
    configs: ``dict``, optional, (default = "{}").
        Additional configs.
    """
    def __init__(self,
                model_file: dict,
                pw: wrapper,
                configs: dict = {}):

        self.pw = pw
        gpu_index = self.pw.auto_device() if 'auto' == configs.get('gpu', 'auto') else int(configs.get('gpu', 'auto'))
        self.device = torch.device("cuda:" + str(gpu_index) if gpu_index >= 0 else "cpu")
        if gpu_index >= 0:
            torch.cuda.set_device(gpu_index)

        name_list = ['gw_map', 'c_map', 'y_map', 'config', 'model']
        gw_map, c_map, y_map, config, model_param = [model_file[tup] for tup in name_list]

        logger.info('Building sequence labeling model.')
        self.seq_model = SeqLabel.from_params(config)
        self.seq_model.load_state_dict(model_param)
        self.seq_model.to(self.device)
        self.seq_model.eval()

        logger.info('Building predictor.')
        self.predictor = predict_wc(self.device, gw_map, c_map, y_map, \
                        label_seq = configs.get("label_seq", "string"), \
                        batch_size = configs.get("batch_size", DEFAULT_BATCH_SIZE), \
                        flm_map = model_file.get("flm_map", None), \
                        blm_map = model_file.get("blm_map", None))

        logger.info('Model is ready.')

    def decode(self, documents):
        """
        Decode documents.

        Parameters
        ----------
        documents: ``list``, required.
            List of str or list of list of str.
        """
        self.seq_model.eval()
        return self.predictor.output_batch(self.seq_model, documents)

def decoder_wrapper(model_file_path: str = "http://dmserv4.cs.illinois.edu/pner0.th", 
                    configs: dict = {}):
    """
    Wrapper for different decode functions.

    Parameters
    ----------
    model_file_path: ``str``, optional, (default = "http://dmserv4.cs.illinois.edu/pner0.th").
        Path to loaded checkpoint.
    configs: ``dict``, optional, (default = "{}").
        Additional configs.
    """
    pw = wrapper(configs.get("log_path", None))

    logger.info("Loading model from {} (might download from source if not cached).".format(model_file_path))
    model_file = wrapper.restore_checkpoint(model_file_path)

    model_type = model_file['config'].get("model_type", 'char-lstm-crf')
    logger.info('Preparing the pre-trained {} model.'.format(model_type))
    model_type_dict = { \
            "char-lstm-crf": decoder_wc,
            "char-lstm-two-level": decoder_tl}
    return model_type_dict[model_type](model_file, pw, configs)

class decode():
    """
    Function for the subcommand.
    """
    def add_subparser(self, name, parser):
        subparser = parser.add_parser(name, description="Decode raw corpus into a file", help='Decode raw corpus')
        subparser.add_argument('-i', '--input_file', type=str, required = True, help="The path to the input file.")
        subparser.add_argument('-o', '--output_file', type=str, required = True, help="The path to the output file.")
        subparser.add_argument('-m', '--model_file', type=str, default="http://dmserv4.cs.illinois.edu/pner0.th", help="Path to pre-trained model")
        subparser.add_argument('-g', '--gpu', type=str, default="auto", help="Device choice (default: 'auto')")
        subparser.add_argument('-d', '--decode_type', choices=['label', 'string'], default='string', help="The type of decoding object")
        subparser.add_argument('-b', '--batch_size', type=int, default=50, help="The size of batch")
        subparser.add_argument('-f', '--file_format', type=str, default="conll", help="The format of input files.")
        subparser.add_argument('--log_path', type=str, default=None, help="The path to the log folder.")
        subparser.set_defaults(func=decode_file)

        return subparser

def decode_file(args):
    """
    Decode file handler function.
    """
    configs = vars(args)

    decoder = decoder_wrapper(configs['model_file'], configs)

    wp = decoder.pw
    logger.info('Loading the corpus.')

    with codecs.open(configs['input_file'], 'r', 'utf-8') as f:
        lines = f.readlines()

    format_handler = {"conll": read_conll_features}
    documents = format_handler[configs.get('file_format', 'conll')](lines)
    
    logger.info('Annotating Documents.')

    with open(configs['output_file'], 'w') as fout:
        detected_entity_inline = decoder.decode(documents)
        for document in detected_entity_inline:
            fout.write('-DOCSTART- -DOCSTART- -DOCSTART-\n\n')
            for sentence in document:
                fout.write(sentence + '\n\n')
