from torch_scope import wrapper
from model_seq.seqlabel import SeqLabel 
from model_seq.predictor import predict_wc

import torch
import numpy as np

# class decoder(object):
#     def __init__(self, model_file_path):


class decoder_wc(object):
    def __init__(self, 
                model_file_path: str, 
                gpu: str = "auto",
                log: str = "./log/",
                label_seq: bool = False,
                batch_size: int = 50):
        self.pw = wrapper(log)
        self.pw.set_level('info')
        gpu_index = self.pw.auto_device() if 'auto' == args.gpu else int(args.gpu)
        self.device = torch.device("cuda:" + str(gpu_index) if gpu_index >= 0 else "cpu")
        if gpu_index >= 0:
            torch.cuda.set_device(gpu_index)

        self.pw.info('Reading the checkpoint from {}'.format(args.model_file))
        model_file = wrapper.restore_checkpoint(args.model_file)
        name_list = ['flm_map', 'blm_map', 'gw_map', 'c_map', 'y_map', 'config', 'model']
        flm_map, blm_map, gw_map, c_map, y_map, config, model_param = [dataset[tup] for tup in name_list ]

        self.pw.info('Building sequence labeling model.')
        self.seq_model = SeqLabel.from_params(config)
        self.seq_model.load_state_dict(model_param)
        self.seq_model.to(device)
        self.seq_model.eval()

        self.pw.info('Building predictor.')
        self.predictor = predict_wc(device, flm_map, blm_map, gw_map, c_map, y_map, label_seq, batch_size)

    def decode(documents):
        return self.predictor(self.seq_model, documents)
