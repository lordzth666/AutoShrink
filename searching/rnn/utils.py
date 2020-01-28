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
import os, shutil
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import pickle
import random


def repackage_hidden(h):
    if isinstance(h, Variable):
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    #print(data.size())
    if args.cuda:
        data = data.cuda()
    return data


def build_emb(corpus, path, ninp):
    with open(path, 'rb') as f:
        glove = pickle.load(f)
    emb = torch.Tensor(len(corpus.dictionary), ninp).data.uniform_(-0.5, 0.5)
    #print(ninp)
    for w, v in glove.items():
        if w in corpus.dictionary.word2idx.keys():
            id = corpus.dictionary.word2idx[w]
            #print(id, w, len(v))
            emb[id] = torch.tensor(v)
    emb = emb.contiguous()
    print('pretrained emb:', len(glove))
    return emb

def get_batch(source, i, bptt, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len])
    return data, target


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def save_checkpoint(model, optimizer, epoch, path, finetune=False):
    if finetune:
        torch.save(model, os.path.join(path, 'finetune_model.pt'))
        torch.save(optimizer.state_dict(), os.path.join(path, 'finetune_optimizer.pt'))
    else:
        torch.save(model, os.path.join(path, 'model.pt'))
        torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer.pt'))
    torch.save({'epoch': epoch+1}, os.path.join(path, 'misc.pt'))


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
        mask = Variable(mask)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1
    X = F.embedding(
        words, masked_embed_weight,
        padding_idx,
        embed.max_norm, embed.norm_type,
        embed.scale_grad_by_freq, embed.sparse
    )
    return X


class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout, self).__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x


def mask2d(B, D, keep_prob, cuda=True):
    m = torch.floor(torch.rand(B, D) + keep_prob) / keep_prob
    m = Variable(m, requires_grad=False)
    if cuda:
        m = m.cuda()
    return m


def inspect_num_edges(mask):
    nodes_queue = [0]
    nodes = np.shape(mask)[0]
    edges = 0
    for i in range(nodes):
        if not i in nodes_queue:
            continue
        for j in range(i+1, nodes):
            if mask[i][j] == 1:
                nodes_queue.append(j)
                edges += 1
    return edges


# 如果没有入边 则continue，目的是删除孤立的顶点
def optimize_mask(mask):
    """
    Remove redundant edges in the original mask.
    :param mask:
    :return:
    """
    mask_shape = np.shape(mask)
    #print(mask_shape)
    nodes = mask_shape[0]

    new_mask = np.zeros_like(mask)

    nodes_queue = [0]
    for d1 in range(nodes):
        if not d1 in nodes_queue:
            continue
        for d2 in range(d1 + 1, nodes):
            if mask[d1][d2] == 1:
                new_mask[d1][d2] = 1
                nodes_queue.append(d2)
    return new_mask


def convert_to_upper_triangle(mask):
    n = mask.shape[0]
    feats = []
    for i in range(n):
        for j in range(i+1,n):
            feats.append(mask[i][j])
    return feats


def stat_num_concats(mask):
    """

    :param mask: Must be optimized.
    :return:
    """
    mask_shape = np.shape(mask)
    nodes = mask_shape[0]

    # Usually final one should count.
    num_concats = 1
    for d1 in range(nodes):
        degree = 0
        for d2 in range(d1 + 1, nodes):
            if mask[d1][d2] == 1:
                degree += 1
        if degree > 1:
            num_concats += 1
    return num_concats


op_name = ['relu', 'sigmoid', 'tanh', 'identity']
def random_choose_ops(nodes):
    ops = []
    for i in range(nodes):
        ops.append(op_name[random.randint(0, len(op_name) - 1)])
    return ops

def get_leaf(mask):
    nodes = len(mask)
    leaf = []
    '''
    for i in range(nodes):
        flag = False
        for j in range(i):
            if mask[j][i]:
                flag = True
                break
        if not flag:
            continue
        for j in range(i + 1, nodes):
            if mask[i][j]:
                flag = False
                break
        if flag:
            leaf.append(i)
    '''
    in_nodes = set()
    in_nodes.add(0)
    for i in range(nodes):
        if i not in in_nodes:
            continue
        for  j in range(i + 1, nodes):
            if mask[i][j] and j not in in_nodes:in_nodes.add(j)
        flag = True
        for j in range(i + 1, nodes):
            if mask[i][j]:
                flag = False
                break
        if flag:leaf.append(i)
    return leaf

