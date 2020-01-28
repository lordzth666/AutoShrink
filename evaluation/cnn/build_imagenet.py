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
import pickle

import sys, os
cwd = os.getcwd()
sys.path.append(cwd)

from evaluation.cnn.imagenet_estimator import ImageNetEstimator
from api.backend import G

G.config_bn_params(1e-3, 0.999)

import argparse

def main(args):
    G.set_conv_triplet(args.triplet)
    with open(args.graph_path, 'rb') as fp:
        meta_graph = pickle.load(fp)

    meta_graph.set_depth(5)

    model_root = args.graph_path

    proto_writer = meta_graph.create_eval_graph(replicate=args.replicate,
                                                scale=args.scale,
                                                channel_scale=args.channel_scale,
                                                task='imagenet-%d' %args.image_size,
                                                model_root=model_root,
                                                pool=args.pool_size,
                                                use_bias=False,
                                                use_residual=args.use_residual,
                                                bottleneck_factor=args.bottleneck_factor)
    proto_writer.set_global_regularization(args.regularizer)
    tf.logging.info("Use %s activation" %args.activation)
    proto_writer.set_activation(args.activation)
    prototxt = "shrinkcnn_demo.prototxt"

    with open(prototxt, 'w') as fp:
        proto_writer.dump(fp)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')

    required.add_argument('--graph_path', type=str, nargs='?',
                          help='Meta-graph path. This argument is required.',
                          default="v3_graph/demo")
    optional.add_argument('--scale', type=float, nargs='*',
                          help="Block Scaling factor. Default is 1.0.",
                          default=[0.5, 1, 2, 4, 8])
    optional.add_argument('--channel_scale', type=float, nargs='?',
                          help="Channel scaling factor",
                          default=1.0)
    optional.add_argument("--replicate", type=int, nargs='*',
                          help="Replicate list",
                          default=[1, 3, 4, 6, 4])
    optional.add_argument('--regularizer', type=float, nargs='?',
                          help="Regularizer strength. Default is 1e-4.",
                          default=5e-5)
    optional.add_argument("--use_residual", action='store_true',
                          help="Whether to use residual connections")
    optional.add_argument("--bottleneck_factor", type=float, nargs='?',
                          help="Bottleneck shrink factor",
                          default=1)
    optional.add_argument("--pool_size", type=int, nargs='*',
                          help="Pool size for each layer",
                          default=[1, 2, 2, 2, 2])
    optional.add_argument("--depth_multiplier", type=float,
                          nargs='?', help="Depth multiplier",
                          default=1.0)
    optional.add_argument('--activation', type=str, nargs='?',
                          help="activation function",
                          default='relu')
    optional.add_argument('--triplet', type=str, nargs='?',
                          help="Conv execution triplet",
                          default='conv-bn-relu')
    optional.add_argument("--image_size", type=int, nargs='?',
                          help="Image size for ImageNet Training",
                          default=224)
    parser._action_groups.append(optional)
    args = parser.parse_args()
    main(args)
