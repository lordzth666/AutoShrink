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

import sys
import os
dir_path = os.getcwd()
sys.path.append(dir_path)

import argparse
from api.backend import G
G.config_bn_params(1e-3, 0.999)


import numpy as np
from evaluation.cnn.imagenet_estimator import ImageNetEstimator

def main(args):
    G.set_conv_triplet(args.triplet)
    estimator= ImageNetEstimator(
        image_size=args.image_size,
        prototxt=args.proto,
        solvertxt=args.solver
    )
    top1_acc, top5_acc, mac = estimator.val(ngpus=args.ngpus,
                                            tfrecord_path=args.tfrecord_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')

    required.add_argument("--proto", required=True,
                          type=str, nargs='?',
                          help="Prototxt name")
    optional.add_argument("--image_size", type=int, nargs='?',
                          help="Image Size",
                          default=224)
    optional.add_argument('--solver', type=str, nargs='?',
                          help="Neural network solver path. Default is ./solver/solver.prototxt.",
                          default="./solver/solver.prototxt")
    optional.add_argument('--ngpus', type=int,
                          help="Number of GPU Slaves",
                          default=1)
    optional.add_argument('--triplet', type=str, nargs='?',
                          help="Conv execution triplet",
                          default='conv-bn-relu')
    optional.add_argument("--tfrecord_path", type=str, nargs='?',
                          help="Path for tfrecord file",
                          default=None)


    parser._action_groups.append(optional)
    args = parser.parse_args()
    print(args)
    main(args)

