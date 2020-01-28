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

import torch
import torch.nn as nn
import torch.nn.functional as F
import searching.rnn.utils
from torch.autograd import Variable
import time
import gc
import torch.nn.functional as F
import math
import numpy as np
import os
INITRANGE = 0.04


class RNN(nn.Module):
    def __init__(self, args, ntoken, emb, nodes):
        super(RNN, self).__init__()
        self.ntoken = ntoken
        self.nodes = nodes
        self.ninp = args.emsize
        self.nhid = args.nhid
        self.nhidlast = args.nhidlast

        self.dropout = args.dropout
        self.dropouth = args.dropouth
        self.dropoutx = args.dropoutx
        self.dropouti = args.dropouti
        self.dropoute = args.dropoute

        #self.bn = nn.BatchNorm1d(self.nhid, affine=False)
        self.lockdrop = utils.LockedDropout()
        self.pretrain = emb
        

        self.emb = nn.Embedding(ntoken, args.emsize)
        self.decoder = nn.Linear(args.emsize, ntoken)
        self.decoder.weight = self.emb.weight
        self.init_weights(emb)
        #self.decoder.weight.data.copy_(self.emb.weight.data)

    def init_weights(self, pretrained_emb):
        #self.emb.weight.data.copy_(pretrained_emb)
        #self.decoder.bias.data.fill_(0)
        #self.emb.weight.requires_grad = False
        #self.decoder.weight.requires_grad = False


        self._W0 = nn.Parameter(torch.Tensor(self.ninp + self.nhid, 2 * self.nhid).uniform_(-INITRANGE, INITRANGE))
        self._Ws = []
        for i in range(self.nodes):
            tmp = [None for _ in range(self.nodes)]
            for j in range(i+1,self.nodes):
                tmp[j] = nn.Parameter(
                    torch.Tensor(self.nhid, 2 * self.nhid).uniform_(-INITRANGE, INITRANGE))
            tmp = nn.ParameterList(tmp)
            self._Ws.append(tmp)
        self._Ws = nn.ModuleList(self._Ws)


    def build_weight(self, mask, leaf_nodes, ops):
        self.emb.weight.data.copy_(self.pretrain)
        assert self.emb.weight is self.decoder.weight
        self.decoder.bias.data.fill_(0)

        self._W0.data = self._W0.data.uniform_(-INITRANGE, INITRANGE)
        for i in range(self.nodes):
            for j in range(i + 1, self.nodes):
                if mask[i][j]:
                    if self._Ws[i][j] is None:
                        self._Ws[i][j] = nn.Parameter(
                            torch.Tensor(self.nhid, 2 * self.nhid).uniform_(-INITRANGE, INITRANGE))
                    else:
                        self._Ws[i][j].data = self._Ws[i][j].data.uniform_(-INITRANGE, INITRANGE)
                else:
                    self._Ws[i][j] = None
        
        self.mask = mask
        self.leaf_nodes = leaf_nodes
        self.ops = ops
        for para in self.parameters():print(para.size())
        total = 0
        for para in self.parameters():
            total += np.prod(para.size())
        print(total)
    def forward(self, input, hidden, return_h=False):
        batch_size = input.size(1)

        emb = utils.embedded_dropout(self.emb, input, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)
        raw_output, new_h = self.cell(emb, hidden)
        output = self.lockdrop(raw_output, self.dropout)
        logit = self.decoder(output.view(-1, self.ninp))
        log_prob = nn.functional.log_softmax(logit, dim=-1)
        model_output = log_prob.view(-1, batch_size, self.ntoken)
        if return_h:
            return model_output, new_h, raw_output, output
        else:
            return model_output, new_h

    def cell(self, inputs, hidden):
        T, B = inputs.size(0), inputs.size(1)
        #print(inputs.size(), hidden.size())
        if self.training:
            x_mask = utils.mask2d(B, inputs.size(2), keep_prob=1. - self.dropoutx)
            h_mask = utils.mask2d(B, hidden.size(2), keep_prob=1. - self.dropouth)
        else:
            x_mask = h_mask = None
        hidden = hidden[0]
        hiddens = []
        for t in range(T):
            hidden = self.node_op(inputs[t], hidden, x_mask, h_mask)
            hiddens.append(hidden)
        hiddens = torch.stack(hiddens)
        return hiddens, hiddens[-1].unsqueeze(0)

    def _compute_init_state(self, x, h_prev, x_mask, h_mask):
        if self.training:
            xh_prev = torch.cat([x * x_mask, h_prev * h_mask], dim=-1)
        else:
            xh_prev = torch.cat([x, h_prev], dim=-1)
        c0, h0 = torch.split(xh_prev.mm(self._W0), self.nhid, dim=-1)
        c0 = c0.sigmoid()
        h0 = h0.tanh()
        prim = h_prev + c0 * (h0 - h_prev)
        return prim

    def _get_activation(self, name):
        #return lambda x: x * F.sigmoid(x)
        if name == 'tanh':
            f = F.tanh
        elif name == 'relu':
            f = F.relu
        elif name == 'sigmoid':
            f = F.sigmoid
        elif name == 'identity':
            f = lambda x: x
        else:
            raise NotImplementedError
        return f
    '''
    def node_op(self, x, h_prev, x_mask, h_mask):
        #if x_mask is None:print('fuck')
        #if h_mask is None:print('shit')
        #print(x.size(), h_prev.size(), x_mask.size(), h_mask.size())
        s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)
        input_state = [None for _ in range(self.nodes)]
        input_state[0] = s0
        for i in range(self.nodes):
            tmp = []
            matri = []
            for j in range(i):
                if input_state[j] is not None and self.mask[j][i] == 1:
                    tmp.append(input_state[j])
                    matri.append(self._Ws[j][i])
            if len(tmp) == 0:continue
            num = len(tmp)
            tmp = torch.cat(tmp, dim=-1)
            matri = torch.cat(matri, dim=0)
            print(num, tmp.size(), matri.size())
            if self.training:
                ch = (tmp * h_mask.expand(-1, tmp.size(1))).mm(matri)
            else:
                ch = tmp.mm(matri)
            ch /= num
            c, h = torch.split(ch, self.nhid, dim=-1)
            c = c.sigmoid()
            fn = self._get_activation(self.ops[i])
            h = fn(h)
            s = s_prev + c * (h - s_prev)
            input_state[i] = s
        output = torch.mean(torch.stack([input_state[i] for i in self.leaf_nodes], -1), -1)
        return output
    '''
    def node_op(self, x, h_prev, x_mask, h_mask):
        s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)
        #s0 = self.bn(s0)
        input_state = [None for _ in range(self.nodes)]
        count = [0 for _ in range(self.nodes)]
        input_state[0] = s0
        
        count[0] = 1
        for i in range(self.nodes):
            if input_state[i] is None:
                continue
            input_state[i] /= count[i]
            #fn = self._get_activation(self.ops[i])
            #input_state[i] = fn(input_state[i])
            s_prev = input_state[i]
            for j in range(i + 1, self.nodes):
                if self.mask[i][j] == 0:
                    continue
                '''
                if i == 0:
                    if self.training:
                        ch = (s_prev * h_mask).mm(self._Ws[i][j])
                    else:
                        ch = s_prev.mm(self._Ws[i][j])
                    c, h = torch.split(ch, self.nhid, dim=-1)
                    c = c.sigmoid()
                    #fn = self._get_activation(self.ops[j])
                    #h = fn(h)
                else:
                    fn = self._get_activation(self.ops[j])
                     
                    tmp = fn(s_prev)
                    #tmp2 = s_prev.sigmoid()
                    if self.training:
                        tmp = tmp * h_mask
                        #tmp2 = tmp2 * h_mask
                    ch = tmp.mm(self._Ws[i][j])
                    c, h = torch.split(ch, self.nhid, dim=-1)
                    c = c.sigmoid()
                '''
                if self.training:
                    ch = (s_prev * h_mask).mm(self._Ws[i][j])
                else:
                    ch = s_prev.mm(self._Ws[i][j])
                c, h = torch.split(ch, self.nhid, dim=-1)
                c = c.sigmoid()
                fn = self._get_activation(self.ops[j])
                h = fn(h)
                s = s_prev + c * (h - s_prev)
                #s = self.bn(s)
                if input_state[j] is None:
                    input_state[j] = s
                else:
                    input_state[j] += s
                count[j] += 1
        output = torch.mean(torch.stack([input_state[i] for i in self.leaf_nodes], -1), -1)
        return output

    def init_hidden(self, size):
        return Variable(torch.zeros(1, size, self.nhid).cuda())


class Estimator:
    def __init__(self, args, ntoken, emb, nodes):
        self.rnn = RNN(args, ntoken, emb, nodes)
        if args.continue_train:
            self.rnn = torch.load(os.path.join(args.save, 'rnn.pt'))
        else:
            self.rnn = RNN(args, ntoken, emb, nodes)

        if args.cuda:
            if args.single_gpu:
                self.rnn = self.rnn.cuda()
            else:
                self.rnn = nn.DataParallel(self.rnn, dim=1).cuda()
        self.cuda = args.cuda
        self.batch_size = args.batch_size
        self.small_batch_size = args.small_batch_size
        self.bptt = args.bptt
        self.alpha = args.alpha
        self.beta = args.beta
        self.clip = args.clip
        self.lr = args.lr
        self.wdecay = args.wdecay
        self.flops = args.nhid * args.nhid

    def compute_flops(self, mask):
        new_mask = utils.optimize_mask(mask.copy())
        return sum([sum(m) for m in new_mask]) * self.flops
    def train_and_eval(self, mask, ops, train_data, val_data, epoch_num, logging):
        leafs = utils.get_leaf(mask)
        self.rnn.build_weight(mask, leafs, ops)
        if self.cuda:
            self.rnn.cuda()
        optimizer = torch.optim.SGD(self.rnn.parameters(), lr=self.lr, weight_decay=self.wdecay)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        best_val_loss = 100000
        failuer = 0
        for epoch in range(epoch_num):
            epoch_start_time = time.time()
            train_loss = self.train(optimizer, train_data)
            logging.info('-' * 89)
            logging.info('| epoch {:3d}  | lr {:02.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], train_loss, math.exp(train_loss)))

            val_loss = self.evaluate(val_data)

            logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            logging.info('-' * 89)

            #scheduler.step()
            if epoch == epoch_num - 4 or epoch == epoch_num - 2:optimizer.param_groups[0]['lr'] /= 10

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                failuer = 0
            else:
                failuer += 1
            if failuer >= 4:
                break

            if val_loss > 3 * train_loss:
                break

            #if epoch == 3 and val_loss > 100: #???????????????????????????????????
            #    break
        return math.exp(best_val_loss) if best_val_loss<30 else math.nan, self.compute_flops(mask)
    def train(self, optimizer, train_data, momentum=0.90):
        moving_loss = 0
        hidden = [self.rnn.init_hidden(self.small_batch_size)
                  for _ in range(self.batch_size // self.small_batch_size)]
        batch, i = 0, 0
        while i < train_data.size(0) - 1 - 1:
            bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
            seq_len = int(bptt)

            lr2 = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = lr2 * seq_len / self.bptt
            self.rnn.train()
            data, targets = utils.get_batch(train_data, i, self.bptt, seq_len=seq_len)

            optimizer.zero_grad()

            start, end, s_id = 0, self.small_batch_size, 0
            while start < self.batch_size:
                cur_data, cur_targets = data[:, start: end], targets[:, start: end].contiguous().view(-1)
                # Starting each batch, we detach the hidden state from how it was previously produced.
                #print(cur_data.size(), hidden[s_id].size())# If we didn't, the model would try backpropagating all the way to start of the dataset.
                hidden[s_id] = utils.repackage_hidden(hidden[s_id])
                log_prob, hidden[s_id], rnn_h, dropped_rnn_h = self.rnn(cur_data, hidden[s_id], return_h=True)
                raw_loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), cur_targets)

                loss = raw_loss
                # Activiation Regularization
                if self.alpha > 0:
                    loss = loss + self.alpha * dropped_rnn_h.pow(2).mean()
                # Temporal Activation Regularization (slowness)
                loss = loss + self.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean()
                loss *= self.small_batch_size / self.batch_size
                moving_loss = raw_loss.data * (1 - momentum) + moving_loss * momentum if moving_loss >0 else raw_loss.data
                loss.backward()
                #print(moving_loss, raw_loss, self.alpha * dropped_rnn_h.pow(2).mean(), self.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean())
                s_id += 1
                start = end
                end = start + self.small_batch_size

                gc.collect()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
            torch.nn.utils.clip_grad_norm(self.rnn.parameters(), self.clip)
            optimizer.step()
            optimizer.param_groups[0]['lr'] = lr2
            batch += 1
            i += seq_len
        return moving_loss.data

    def evaluate(self, val_data):
        # Turn on evaluation mode which disables dropout.
        self.rnn.eval()
        total_loss = 0
        hidden = self.rnn.init_hidden(val_data.size(1))
        for i in range(0, val_data.size(0) - 1, self.bptt):
            data, targets = utils.get_batch(val_data, i, self.bptt, evaluation=True)
            targets = targets.view(-1)
            #print(data.size(), hidden.size())
            log_prob, hidden = self.rnn(data, hidden)
            loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets).data
            total_loss += loss.data * len(data)

            hidden = utils.repackage_hidden(hidden)
        return total_loss.data / len(val_data)
