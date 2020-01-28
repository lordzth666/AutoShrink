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

import numpy as np
import argparse
import os, sys, glob

sys.path.append(os.getcwd())

import time
import numpy as np
import torch
import logging
import torch.nn as nn
import torch.backends.cudnn as cudnn
from evaluation.rnn.model import RNN
import data
from evaluation.rnn.utils import batchify, get_batch, repackage_hidden, create_exp_dir, save_checkpoint, build_emb
import evaluation.rnn.utils as utils
import gc
import math


def add_parse():
    parser = argparse.ArgumentParser(description='PyTorch PennTreeBank/WikiText2 Language Model')
    parser.add_argument('--data', type=str, default='./data/penn/',
                        help='location of the data corpus')
    parser.add_argument('--emb_path', type=str, default='./w2v.pkl',
                        help='location of the pretrained embeddings')
    parser.add_argument('--nodes', type=int, default=6,
                        help='nodes num')
    parser.add_argument('--emsize', type=int, default=745,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=745,
                        help='number of hidden units per layer')
    parser.add_argument('--nhidlast', type=int, default=745,
                        help='number of hidden units for the last rnn layer')
    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=3000,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.75,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropouth', type=float, default=0.25,
                        help='dropout for hidden nodes in rnn layers (0 = no dropout)')
    parser.add_argument('--dropoutx', type=float, default=0.75,
                        help='dropout for input nodes in rnn layers (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0.2,
                        help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropoute', type=float, default=0.1,
                        help='dropout to remove words from embedding layer (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=1267,
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
    parser.add_argument('--beta', type=float, default=1e-3,
                        help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
    parser.add_argument('--wdecay', type=float, default=8e-7,
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




def train(optimizer, train_data, momentum=0.97):
    model.train()
    moving_loss = 0
    hidden = [model.init_hidden(args.small_batch_size)
              for _ in range(args.batch_size // args.small_batch_size)]
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        seq_len = min(seq_len, args.bptt + args.max_seq_len_delta)



        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = utils.get_batch(train_data, i, args.bptt, seq_len=seq_len)

        optimizer.zero_grad()

        start, end, s_id = 0, args.small_batch_size, 0
        while start < args.batch_size:
            cur_data, cur_targets = data[:, start: end], targets[:, start: end].contiguous().view(-1)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden[s_id] = utils.repackage_hidden(hidden[s_id])
            log_prob, hidden[s_id], rnn_h, dropped_rnn_h = model(cur_data, hidden[s_id], return_h=True)
            raw_loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), cur_targets)

            loss = raw_loss
            # Activiation Regularization
            if args.alpha > 0:
                loss = loss + args.alpha * dropped_rnn_h.pow(2).mean()
            # Temporal Activation Regularization (slowness)
            loss = loss + args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean()
            loss *= args.small_batch_size / args.batch_size
            moving_loss = raw_loss.data * (1 - momentum) + moving_loss * momentum
            loss.backward()

            s_id += 1
            start = end
            end = start + args.small_batch_size

            gc.collect()
        if np.isnan(moving_loss.cpu()):
            raise NotImplementedError
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()
        optimizer.param_groups[0]['lr'] = lr2
        batch += 1
        i += seq_len
    return moving_loss

def evaluate(val_data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, val_data.size(0) - 1, args.bptt):
        data, targets = utils.get_batch(val_data, i, args.bptt, evaluation=True)
        targets = targets.view(-1)

        log_prob, hidden = model(data, hidden)
        loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets).data
        total_loss += loss * len(data)

        hidden = utils.repackage_hidden(hidden)
    return total_loss / len(val_data)


parser = add_parse()
args = parser.parse_args()



INITRANGE = 0.04

#mask = [[0 for _ in range(args.nodes)] for _ in range(args.nodes)]
'''
for i in range(args.nodes):
    for j in range(i + 1, args.nodes):
        mask[i][j] = 1
'''
'''
mask = [[0., 1., 0., 0., 0., 0., 0., 1.],
       [0., 0., 1., 0., 1., 1., 0., 0.],
       [0., 0., 0., 1., 0., 1., 1., 1.],
       [0., 0., 0., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 1., 1., 1.],
       [0., 0., 0., 0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.]]

'''
'''
mask = [[0., 1., 0., 1., 1., 0.],
       [0., 0., 0., 1., 1., 1.],
       [0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 1., 1.],
       [0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0.]]
'''
'''
mask = [[0., 0., 1., 1., 1., 1.],
       [0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 1.],
       [0., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0.]]
'''
'''
mask = [[0., 0., 1., 1., 1., 1.],
       [0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 1., 1.],
       [0., 0., 0., 0., 1., 1.],
       [0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0.]]
'''
mask = [[0., 1., 1., 1., 1., 1.],
       [0., 0., 0., 1., 1., 1.],
       [0., 0., 0., 0., 1., 1.],
       [0., 0., 0., 0., 1., 1.],
       [0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0.]]
ops = ['tanh', 'identity', 'sigmoid', 'sigmoid', 'tanh', 'identity']

#ops = ['sigmoid', 'identity', 'identity', 'tanh', 'relu', 'tanh']
#ops = ['relu', 'relu', 'tanh', 'identity', 'identity', 'sigmoid', 'sigmoid', 'relu']
#ops = ['relu', 'identity', 'sigmoid', 'relu', 'tanh', 'sigmoid']
if args.nhidlast < 0:
    args.nhidlast = args.emsize
if args.small_batch_size < 0:
    args.small_batch_size = args.batch_size
if not args.continue_train:
    args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
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
#emb = build_emb(corpus, args.emb_path, args.emsize)
emb = torch.Tensor(len(corpus.dictionary), args.emsize).data.uniform_(-INITRANGE, INITRANGE)
eval_batch_size = 10

train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
ntokens = len(corpus.dictionary)



if args.continue_train:
    model = torch.load(os.path.join(args.save, 'model.pt'))
else:
    model = RNN(args, ntokens, emb, args.nodes)


model.dropout = args.dropout
model.dropouth = args.dropouth
model.dropouti = args.dropouti
model.dropoute = args.dropoute
model.dropoutx = args.dropoutx

leafs = utils.get_leaf(mask)
if not args.continue_train:
    model.build_weight(mask, leafs, ops)
if args.cuda:
    model.cuda()
if args.continue_train:
    optimizer_state = torch.load(os.path.join(args.save, 'optimizer.pt'))
    if 't0' in optimizer_state['param_groups'][0]:
        optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    optimizer.load_state_dict(optimizer_state)
    print(optimizer.param_groups[0]['lr'])
    optimizer.param_groups[0]['lr'] = args.lr
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10,
#                                                       verbose=True, mode='min', factor=0.1)
best_val_loss = []
stored_loss = 10000
epoch = 0
while epoch < args.epochs:
    epoch_start_time = time.time()
    try:
        train_loss = train(optimizer, train_data)
    except:
        model = torch.load(os.path.join(args.save, 'model.pt'))
        model.cuda()
        optimizer_state = torch.load(os.path.join(args.save, 'optimizer.pt'))
        if 't0' in optimizer_state['param_groups'][0]:
            optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
        optimizer.load_state_dict(optimizer_state)

        epoch = torch.load(os.path.join(args.save, 'misc.pt'))['epoch']
        continue
    logging.info('-' * 89)
    ppl = math.exp(train_loss) if train_loss < 20 else 0
    logging.info('| epoch {:3d}  | lr {:02.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
        epoch, optimizer.param_groups[0]['lr'], train_loss, ppl))



    if 't0' in optimizer.param_groups[0]:
        tmp = {}
        for prm in model.parameters():
            tmp[prm] = prm.data.clone()
            if 'ax' in optimizer.state[prm].keys():
                prm.data = optimizer.state[prm]['ax'].clone()

        val_loss2 = evaluate(val_data)
        ppl = math.exp(val_loss2) if val_loss2 < 20 else 0
        logging.info('-' * 89)
        logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                          val_loss2, ppl))
        logging.info('-' * 89)

        if val_loss2 < stored_loss:
            save_checkpoint(model, optimizer, epoch, args.save)
            logging.info('Saving Averaged!')
            stored_loss = val_loss2

        for prm in model.parameters():
            prm.data = tmp[prm].clone()

    else:
        val_loss = evaluate(val_data)
        ppl = math.exp(val_loss) if val_loss < 20 else 0
        logging.info('-' * 89)
        logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, ppl))
        logging.info('-' * 89)

        if val_loss < stored_loss:
            save_checkpoint(model, optimizer, epoch, args.save)
            logging.info('Saving Normal!')
            stored_loss = val_loss

        if 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
            logging.info('Switching!')
            optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
        best_val_loss.append(val_loss)
        #print(len(best_val_loss), args.nonmono, val_loss, best_val_loss[:-args.nonmono], best_val_loss)
    epoch += 1 
