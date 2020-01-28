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
    G.set_initializer("default")
    estimator= ImageNetEstimator(
        image_size=args.image_size,
        prototxt=args.proto,
        solvertxt=args.solver
    )
    top1_acc, top5_acc, mac = estimator.trainval(train_batch_size=args.batch_size,
                                                 ngpus=args.ngpus,
                                                 save_model=args.save_model,
                                                 epochs=args.epochs,
                                                 max_steps=args.max_steps,
                                                 tfrecord_path=args.tfrecord_path)
    if args.log_dir is not None:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        proto_name = args.proto.split("/")[-1]
        log_name = os.path.join(args.log_dir, proto_name)
        with open(log_name, 'w') as fp:
            fp.write("name: %s\n" % args.proto)
            fp.write("MillonMACs: %s\n" % mac)
            fp.write("Top1Acc: %s\n" % top1_acc)
            fp.write("Top5Acc: %s\n" % top5_acc)
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')

    required.add_argument("--proto", required=True,
                          type=str, nargs='?',
                          help="Prototxt name")
    optional.add_argument('--batch_size', type=int, nargs='?',
                          help='Training batch size, default is 64.',
                          default=128)
    optional.add_argument("--image_size", type=int, nargs='?',
                          help="Image Size",
                          default=224)
    optional.add_argument('--epochs', type=int, nargs='?',
                          help="Training epochs, default is 250.",
                          default=250)
    optional.add_argument('--solver', type=str, nargs='?',
                          help="Neural network solver path. Default is ./solver/solver.prototxt.",
                          default="./solver/solver.prototxt")
    optional.add_argument('--ngpus', type=int,
                          help="Number of GPU Slaves",
                          default=1)
    optional.add_argument("--log_dir", type=str, nargs='?',
                          help="Output log path",
                          default=None)
    optional.add_argument("--save_model", type=bool, nargs='?',
                          help="Whether to save model",
                          default=True)
    optional.add_argument("--max_steps", type=int, nargs="?",
                          help="Max steps to train",
                          default=None)
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

