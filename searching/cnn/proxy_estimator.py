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
from api.network.model import Model
from math import ceil
from api.dataset.preprocess_tfrecord import *
import time
import os, sys
import datetime
from tqdm import tqdm

class ProxyEstimator:
    def __init__(self,
                 prototxt=None,
                 solvertxt=None,
                 verbose=-1,
                 image_size=32):
        self.prototxt = prototxt
        self.solvertxt = solvertxt
        self.verbose = verbose
        self.image_size = image_size

    def trainval_on_proxy(self,
                          train_batch_size=-1,
                          val_batch_size=-1,
                          epochs=-1,
                          ngpus=-1):
        raise BaseException


class CIFARProxyEstimator(ProxyEstimator):
    """
    Estimator on the proxy dataset.
    """
    def __init__(self,
                 prototxt,
                 solvertxt,
                 verbose=1,
                 image_size=32):
        """
        Evaluate the compiled model instance on the proxy dataset.
        :param model:
        """
        super().__init__(prototxt,
                         solvertxt,
                         verbose,
                         image_size)

    def trainval_on_proxy(self,
                          train_batch_size=128,
                          val_batch_size=100,
                          epochs=10,
                          ngpus=1):
        """
        Train and evaluation on proxy dataset.
        :return:
        """

        training_batch_size = train_batch_size * ngpus
        validation_batch_size = val_batch_size * ngpus

        tf_record_list_train = ["./data/CIFAR-10-proxy-train.tfrecord"]
        tf_record_list_val = ["./data/CIFAR-10-proxy-val.tfrecord"]

        num_train_examples = total_num_records(tf_record_list_train)
        num_val_examples = total_num_records(tf_record_list_val)

        num_train_steps = ceil(num_train_examples / training_batch_size)
        num_val_steps = ceil(num_val_examples / validation_batch_size)

        with tf.device("/cpu:0"):
            train_data = tf.data.TFRecordDataset(tf_record_list_train,
                                                 buffer_size=None,
                                                 num_parallel_reads=16)
            train_data = train_data.apply(tf.data.experimental.shuffle_and_repeat(2000))
            train_data = train_data.apply(tf.data.experimental.map_and_batch(map_func=extract_cifar10_raw_fn,
                                                                             batch_size=training_batch_size,
                                                                             drop_remainder=True,
                                                                             num_parallel_calls=16))
            train_data = train_data.prefetch(-1)

            train_iterator = train_data.make_one_shot_iterator()
            train_next_step = []
            for i in range(ngpus):
                train_next_step.append(train_iterator.get_next())

            val_data = tf.data.TFRecordDataset(tf_record_list_val,
                                               buffer_size=None,
                                               num_parallel_reads=16)
            val_data = val_data.repeat(-1)
            val_data = val_data.apply(tf.data.experimental.map_and_batch(
                map_func=extract_cifar10_raw_fn,
                batch_size=validation_batch_size,
                drop_remainder=True,
                num_parallel_calls=16))
            val_data = val_data.prefetch(-1)

            val_iterator = val_data.make_one_shot_iterator()

            val_next_step = []
            for i in range(ngpus):
                val_next_step.append(val_iterator.get_next())

        model = Model(self.prototxt, self.solvertxt, ngpus=ngpus, data_train_op=train_next_step, data_val_op=val_next_step,
                      verbose=self.verbose)
        if self.verbose > 2:
            model.summary()

        best_val_acc = 0
        failure = 0

        for i in range(epochs):
            moving_train_acc = 0.
            moving_train_loss = 0.
            momentum = 0.99
            print("Epoch %d" % i)

            model.train()
            for j in tqdm(range(num_train_steps)):
                loss, acc = model.train_on_batch()
                acc = acc[0]
                if moving_train_acc == 0:
                    moving_train_acc = acc
                else:
                    moving_train_acc = momentum * moving_train_acc + (1 - momentum) * acc
                if moving_train_loss == 0:
                    moving_train_loss = loss
                else:
                    moving_train_loss = momentum * moving_train_loss + (1 - momentum) * loss
                if moving_train_acc > 0.98:
                    break
            print("Train acc: %.5f, Train loss: %.5f" % (moving_train_acc, moving_train_loss))

            model.eval()
            val_loss = 0
            val_acc = 0
            for j in range(num_val_steps):
                loss, acc = model.test_on_batch()
                acc = acc[0]
                val_loss += loss
                val_acc += acc
            val_loss /= num_val_steps
            val_acc /= num_val_steps
            print("Val acc: %5f, Val loss: %.5f" % (val_acc, val_loss))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                failure = 0
            else:
                failure += 1
            if failure == 5:
                break
            # Kill the models with slow start
            if i == 3 and val_acc < 0.2:
                break
            # Early stopping
            if val_loss > moving_train_loss * 3:
                break

        model.close()

        return best_val_acc, model.flops

        pass

