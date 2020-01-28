"""
Copyright (c) <2019> <CEI Lab, Duke University>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
import os, sys, glob
import time
import numpy as np
import torch
import logging
import torch.nn as nn
import torch.backends.cudnn as cudnn

cwd = os.getcwd()
sys.path.append(cwd)

from searching.rnn.model import Estimator
import data
import searching.rnn.autoshrink_rnn as autoshrink
from searching.rnn.utils import batchify, get_batch, repackage_hidden, create_exp_dir, save_checkpoint, build_emb
import pickle


def add_parse():
    parser = argparse.ArgumentParser(description='PyTorch PennTreeBank/WikiText2 Language Model')
    parser.add_argument('--data', type=str, default='./data/penn_proxy',
                        help='location of the data corpus')
    parser.add_argument('--nodes', type=int, default=6,
                        help='num of nodes')
    parser.add_argument('--emb_path', type=str, default='./w2v.pkl',
                        help='location of the pretrained embeddings')
    parser.add_argument('--emsize', type=int, default=100,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=100,
                        help='number of hidden units per layer')
    parser.add_argument('--nhidlast', type=int, default=100,
                        help='number of hidden units for the last rnn layer')
    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=10,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0,#0.75,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropouth', type=float, default=0,#0.25,
                        help='dropout for hidden nodes in rnn layers (0 = no dropout)')
    parser.add_argument('--dropoutx', type=float, default=0,#0.75,
                        help='dropout for input nodes in rnn layers (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0,#0.2,
                        help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropoute', type=float, default=0,#0.1,
                        help='dropout to remove words from embedding layer (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=189,
                        help='random seed')
    parser.add_argument('--nonmono', type=int, default=5,
                        help='random seed')
    parser.add_argument('--cuda', action='store_false',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str,  default='EXP',
                        help='path to save the final model')
    parser.add_argument('--alpha', type=float, default=0,
                        help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
    parser.add_argument('--beta', type=float, default=0,#1e-3,
                        help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
    parser.add_argument('--wdecay', type=float, default=0,#5e-7,
                        help='weight decay applied to all weights')
    parser.add_argument('--continue_train', action='store_true',
                        help='continue train from a checkpoint')
    parser.add_argument('--small_batch_size', type=int, default=-1,
                        help='the batch size for computation. batch_size should be divisible by small_batch_size.\
                         In our implementation, we compute gradients with small_batch_size multiple times, and accumulate the gradients\
                         until batch_size is reached. An update step is then performed.')
    parser.add_argument('--max_seq_len_delta', type=int, default=20,
                        help='max sequence length')
    parser.add_argument('--single_gpu', default=True, action='store_false',
                        help='use single GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')
    return parser


def train(args):
    if args.nhidlast < 0:
        args.nhidlast = args.emsize
    if args.small_batch_size < 0:
        args.small_batch_size = args.batch_size
    if not args.continue_train:
        args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
        create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.set_device(args.gpu)
            cudnn.benchmark = True
            cudnn.enabled = True
            torch.cuda.manual_seed_all(args.seed)

    corpus = data.Corpus(args.data)
    emb = build_emb(corpus, args.emb_path, args.emsize)
    eval_batch_size = 256
    train_data = batchify(corpus.train, args.batch_size, args)
    val_data = batchify(corpus.valid, eval_batch_size, args)
    ntokens = len(corpus.dictionary)
    if args.continue_train:
        with open(os.path.join(args.save, 'model.pt'), 'rb') as f:
            model = pickle.load(f)
            model.logging = logging
    else:
        model = autoshrink.AutoShrinkRNN(args, ntokens, logging, emb, nodes=args.nodes)
    estimator = Estimator(args, ntokens, emb, args.nodes)
    model.auto_shrink(estimator, train_data, val_data, keep_fraction=0.25)






if __name__ == '__main__':
    parser = add_parse()
    args = parser.parse_args()
    train(args)
