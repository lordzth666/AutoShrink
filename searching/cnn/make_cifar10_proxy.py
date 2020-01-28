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

import tensorflow as tf
import numpy as np
import requests
from tqdm import tqdm

import argparse

def RealSubSet(X, y, item_each_cls=500, classes=10):
    total_indices = np.asarray([], dtype=np.int32)

    for i in range(classes):
        indices, _ = np.where(y == i)
        indices_choice = np.random.choice(indices, item_each_cls)
        total_indices = np.append(total_indices, indices_choice, axis=0)

    np.random.shuffle(total_indices)
    return X[total_indices].copy(), y[total_indices].copy()


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def Write_CIFAR_to_tfrecord(X, y, record_dir):
    print(record_dir)
    fp_record = tf.python_io.TFRecordWriter(record_dir)
    print(np.shape(X))
    num_examples = np.shape(X)[0]
    for i in tqdm(range(num_examples)):
        img_raw = X[i].tostring()
        label = y[i]
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'label': _int64_feature(label),
                    'image_raw': _bytes_feature(img_raw)
                }))
        fp_record.write(example.SerializeToString())
    fp_record.close()


def main(args):
    data_train, data_val = tf.keras.datasets.cifar10.load_data()

    X_tr, y_tr = data_train
    X_val, y_val = data_val

    X_sub_tr, y_sub_tr = RealSubSet(X_tr, y_tr, item_each_cls=500)
    X_sub_val, y_sub_val = RealSubSet(X_val, y_val, item_each_cls=100)

    record_dir = args['record_dir']
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)

    train_record_path = os.path.join(record_dir, 'CIFAR-10-proxy-train.tfrecord')
    val_record_path = os.path.join(record_dir, 'CIFAR-10-proxy-val.tfrecord')

    Write_CIFAR_to_tfrecord(X_sub_tr, y_sub_tr, train_record_path)
    Write_CIFAR_to_tfrecord(X_sub_val, y_sub_val, val_record_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument("--record_dir", required=True, type=str, default=None, help="CIFAR-10 TFRecord dir")
    parser._action_groups.append(optional)
    args = vars(parser.parse_args())
    main(args)
