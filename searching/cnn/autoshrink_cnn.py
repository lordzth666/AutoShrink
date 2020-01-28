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

import tensorflow as tf
import os, sys

cwd = os.getcwd()
sys.path.append(cwd)

import pickle
import argparse
from searching.cnn.metagraph import MetaGraph
from api.backend import G

tf.logging.set_verbosity(tf.logging.WARN)

import numpy as np

np.random.seed(5)

def main(args):
    G.set_conv_triplet(args.triplet)
    if args.pretrained_path is None:
        meta_graph = MetaGraph(nodes=args.nodes,
                               p=args.prob,
                               model_root=args.model_root,
                               protocol=args.protocol,
                               solvertxt=args.solver)
    else:
        with open(args.pretrained_path, 'rb') as fp:
            meta_graph = pickle.load(fp)


    meta_graph.auto_shrink(max_steps_to_action=10, keep_fraction=0.01, drop_k_each_iteration=1,
                           scale=args.scale,
                           replicate=args.replicate,
                           pool=args.pool
                           )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", type=str, nargs='?', help="Pretrained graph path",
                        default=None)
    parser.add_argument("--nodes", type=int, nargs='?', help="Number of nodes",
                        default=8)
    parser.add_argument("--depth", type=int, nargs='?', help="Depth in the graph",
                        default=1)
    parser.add_argument("--width", type=int, nargs='?', help="Width in the graph",
                        default=3)
    parser.add_argument("--prob", type=float, nargs='?', help="Connection probability",
                        default=1.0)
    parser.add_argument('--model_root', type=str, nargs='?', help="Model root to save",
                        default="demo/")
    parser.add_argument('--scale', type=float, nargs='*',
                         help="Block Scaling factor. Default is 1.0.",
                         default=[1, 1, 1])
    parser.add_argument("--pool",   type=int, nargs='*',
                          help="Pool size",
                          default=[1, 2, 2])
    parser.add_argument('--channel_scale', type=float, nargs='?',
                          help="Channel scaling factor",
                          default=1)
    parser.add_argument("--replicate", type=int, nargs='*',
                          help="Replicate list",
                          default=[1, 1, 1])
    parser.add_argument("--protocol", type=str, nargs='?',
                        help="Protocol to use",
                        default='v3')
    parser.add_argument("--max_steps", type=int, nargs='?',
                        help="Max steps to action, corresponding to 'k' in the paper.",
                        default=10)
    parser.add_argument('--triplet', type=str, nargs='?',
                          help="Conv execution triplet",
                          default='conv-bn-relu')
    parser.add_argument("--solver", type=str, nargs='?',
                        help="Solver to use during search",
                        default="cifar10_proxy.solver")
    args = parser.parse_args()
    print(vars(args))
    main(args)
