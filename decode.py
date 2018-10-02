from __future__ import print_function
import argparse

from model_seq.decoder import decoder_wc
from model_seq.utils import read_features 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluating LM-BLSTM-CRF')
    parser.add_argument('--model_file', default='./checkpoint/pner0.th')
    parser.add_argument('--log_folder', default='./log/ner/')
    parser.add_argument('--gpu', type=str, default="auto")
    parser.add_argument('--decode_type', choices=['label', 'string'], default='string')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--input_file', default='data/ner2003/test.txt')
    parser.add_argument('--output_file', default='output.txt')
    args = parser.parse_args()

    decoder = decoder_wc(self.model_file, self.gpu, self.log_folder, self.decode_type == 'label', args.batch_size)

    wp = decoder.pw
    wp.info('Loading the corpus')

    with codecs.open(args.input_file, 'r', 'utf-8') as f:
        lines = f.readlines()

    # converting format
    documents = read_features(lines)
    
    print('annotating')
    with open(args.output_file, 'w') as fout:
        fout.write(decoder.decode(documents))
