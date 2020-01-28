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
import sys, os

from api.network.model import Model
from math import ceil
from api.dataset.preprocess_tfrecord import *
from api.dataset.preprocess_tfrecord import _extract_imagenet_fn, _extract_imagenet_raw_fn

from evaluation.cnn.base_estimator import BaseEstimator
import time
import datetime

_supported_imagenet_sizes = [32, 96, 128, 160, 192, 224]

def extract_imagenet_fn_mux(image_size):
    if image_size in _supported_imagenet_sizes:
        return lambda tfrecord: _extract_imagenet_fn(tfrecord, image_size)
    else:
        raise NotImplementedError


def extract_imagenet_raw_fn_mux(image_size):
    if image_size in _supported_imagenet_sizes:
        return lambda tfrecord: _extract_imagenet_raw_fn(tfrecord, image_size)
    else:
        raise NotImplementedError


class ImageNetEstimator(BaseEstimator):
    """
    Estimator on the CIFAR-10 Dataset
    """
    def __init__(self,
                 prototxt,
                 solvertxt,
                 verbose=1,
                 image_size=224):
        """
        Evaluate the compiled model instance on the proxy dataset.
        :param model:
        """
        super().__init__(prototxt,
                         solvertxt,
                         verbose,
                         image_size)
        self.prototxt = prototxt
        self.solvertxt = solvertxt
        self.verbose = verbose
        self.image_size = image_size
        pass

    def trainval(self,
                 train_batch_size=128,
                 val_batch_size=100,
                 epochs=350,
                 ngpus=1,
                 warmup_factor=1.0,
                 warmup_epochs=-1,
                 save_model=True,
                 max_steps=None,
                 tfrecord_path=None):

        """
        Train and evaluation on ImageNet dataset.
        :return:
        """
        # Declare constants at the head
        PREFETCH_DATASET_BUFFER_SIZE = None
        NUM_FILES_FEED = 16
        FOLLOWUP_SHUFFLE_BUFFER_SIZE = 8192
        NUM_PARALLEL_CALLS = 16
        NUM_BLOCK_LENGTH = 1
        NUM_SHARDS = 16
        PREFETCH_SIZE = -1

        num_train_examples = 1281167
        print("Training dataset has %d examples." % num_train_examples)
        # num_val_examples = total_num_records(tf_record_list_val)
        num_val_examples = 50000
        print("Validation dataset has %d examples." % num_val_examples)

        def prefetch_dataset(filename):
            dataset = tf.data.TFRecordDataset(
                filename, buffer_size=PREFETCH_DATASET_BUFFER_SIZE,
            num_parallel_reads=NUM_PARALLEL_CALLS)
            return dataset

        training_batch_size = train_batch_size
        validation_batch_size = val_batch_size

        print("Use %d GPUs to train..." %ngpus)
        print("Mini-batch-size: %d" %(train_batch_size))

        num_train_steps = ceil(num_train_examples / training_batch_size / ngpus)
        num_val_steps = ceil(num_val_examples / validation_batch_size / ngpus)

        extract_imagenet_fn = extract_imagenet_fn_mux(image_size=self.image_size)
        extract_imagenet_raw_fn = extract_imagenet_raw_fn_mux(image_size=self.image_size)

        image_size_ = 256

        if tfrecord_path is None:
            train_tfrecord_path = os.path.join("./data/ImageNet/TFRecords/ILSVRC%d/train" % (image_size_))
            val_tfrecord_path = os.path.join("./data/ImageNet/TFRecords/ILSVRC%d/val" % (image_size_))
            train_file_patten = os.path.join(train_tfrecord_path, 'imagenet-train-*')
            val_file_patten = os.path.join(val_tfrecord_path, 'imagenet-val-*')
        else:
            train_tfrecord_path = os.path.join(tfrecord_path, "train_tfrecord")
            val_tfrecord_path = os.path.join(tfrecord_path, "val_tfrecord")
            train_file_patten = os.path.join(train_tfrecord_path, 'train-*')
            val_file_patten = os.path.join(val_tfrecord_path, 'validation-*')

        print("training path: %s" %train_tfrecord_path)

        with tf.device("/cpu:0"):
            train_data = tf.data.Dataset.list_files(train_file_patten, shuffle=True)
            train_data = train_data.repeat(-1)
            train_data = train_data.apply(
                tf.data.experimental.parallel_interleave(
                    prefetch_dataset,
                    cycle_length=NUM_FILES_FEED,
                    sloppy=True,
                    block_length=NUM_BLOCK_LENGTH,
                ))

            train_data = train_data.apply(tf.data.experimental.shuffle_and_repeat(FOLLOWUP_SHUFFLE_BUFFER_SIZE))

            train_data = train_data.apply(tf.data.experimental.map_and_batch(map_func=extract_imagenet_fn,
                                                                             batch_size=training_batch_size,
                                                                             num_parallel_calls=NUM_PARALLEL_CALLS,
                                                                             drop_remainder=True))

            train_data = train_data.prefetch(PREFETCH_SIZE)

            train_iterator = train_data.make_one_shot_iterator()
            train_next_step = []
            for i in range(ngpus):
                train_next_step.append(train_iterator.get_next())

            val_data = tf.data.Dataset.list_files(val_file_patten, shuffle=False)

            val_data = val_data.apply(
                tf.data.experimental.parallel_interleave(
                    prefetch_dataset,
                    cycle_length=1,
                    sloppy=False))

            val_data = val_data.repeat(-1)

            val_data = val_data.apply(tf.data.experimental.map_and_batch(
                map_func=extract_imagenet_raw_fn,
                drop_remainder=True,
                batch_size=validation_batch_size,
                num_parallel_calls=10))

            val_iterator = val_data.make_one_shot_iterator()

            val_next_step = []
            for i in range(ngpus):
                val_next_step.append(val_iterator.get_next())

        model = Model(self.prototxt, self.solvertxt, ngpus=ngpus, data_train_op=train_next_step,
                      data_val_op=val_next_step)
        model.summary()


        save_path = model.save_path

        val_top1_accuracy = None
        val_top1_accuracy_summary = tf.Summary()
        val_top1_accuracy_summary.value.add(tag='val_top1_accuracy', simple_value=val_top1_accuracy)

        val_top5_accuracy = None
        val_top5_accuracy_summary = tf.Summary()
        val_top5_accuracy_summary.value.add(tag='val_top5_accuracy', simple_value=val_top5_accuracy)

        val_loss = None
        val_loss_summary = tf.Summary()
        val_loss_summary.value.add(tag='val_loss', simple_value=val_loss)

        best_val_acc = 0
        best_val5_acc = 0

        model.save_to_graphpb(model.model_root)

        global_steps = 0
        total_loss = 0
        total_top1_acc = 0
        total_top5_acc = 0

        total_epochs = model.global_epochs+epochs
        for i in range(model.global_epochs, total_epochs):
            print("Epoch %d/%d" % (i + 1, total_epochs))

            start = time.time()
            model.train()

            for t_steps in range(num_train_steps):
                global_steps += ngpus
                loss, acc = model.train_on_batch()
                if total_loss == 0:
                    total_loss = loss
                    total_top1_acc = acc[0]
                    total_top5_acc = acc[1]
                else:
                    total_loss = total_loss * 0.99 + loss * 0.01
                    total_top1_acc = total_top1_acc * 0.99 + acc[0] * 0.01
                    total_top5_acc = total_top5_acc * 0.99 + acc[1] * 0.01
                if global_steps % 32 == 0:
                    end = time.time()
                    num_example_per_sec = training_batch_size / (end - start) * 32
                    print(datetime.datetime.now())
                    print("Step %d: loss=%.5f, acc@1=%.5f, acc@5=%.5f (%.2f examples per second)"
                                    % (global_steps, total_loss,
                                       total_top1_acc,
                                       total_top5_acc,
                                       num_example_per_sec))
                    start = time.time()

                if max_steps is not None:
                    if global_steps > max_steps:
                        break

            average_loss = total_loss
            average_top1_acc = total_top1_acc
            average_top5_acc = total_top5_acc

            print("Train acc@1: %5f, Train acc@5: %.5f, Train loss: %.5f"
                            % (average_top1_acc, average_top5_acc, average_loss))

            if save_model:
                print("Model saved at step %d" %global_steps)
                model.saver.save(model.sess, save_path, global_step=global_steps)

            # Doing validation
            val_loss = 0
            val_acc = 0
            val_top5_acc = 0

            model.eval()

            for j in range(num_val_steps):
                loss, acc = model.test_on_batch()
                val_loss += loss
                val_acc += acc[0]
                val_top5_acc += acc[1]

            val_loss /= num_val_steps
            val_acc /= num_val_steps
            val_top5_acc /= num_val_steps

            print("Val acc@1: %5f, Val acc@5: %.5f, Val loss: %.5f" % (val_acc, val_top5_acc, val_loss))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val5_acc = val_top5_acc

            val_top1_accuracy_summary.value[0].simple_value = val_acc
            model.summary_writer.add_summary(val_top1_accuracy_summary, i)

            val_top5_accuracy_summary.value[0].simple_value = val_top5_acc
            model.summary_writer.add_summary(val_top5_accuracy_summary, i)

            val_loss_summary.value[0].simple_value = val_loss
            model.summary_writer.add_summary(val_loss_summary, i)

            if max_steps is not None:
                if global_steps == max_steps:
                    break

        model.close()
        return best_val_acc, best_val5_acc, float(model.flops) / 1e6


    def val(self,
            val_batch_size=100,
            ngpus=1,
            tfrecord_path=None):

        PREFETCH_DATASET_BUFFER_SIZE = None
        NUM_FILES_FEED = 128
        FOLLOWUP_SHUFFLE_BUFFER_SIZE = 2000
        NUM_PARALLEL_CALLS = -1
        NUM_BLOCK_LENGTH = 1
        NUM_SHARDS = 128

        num_val_examples = 50000
        print("Validation dataset has %d examples." % num_val_examples)

        def prefetch_dataset(filename):
            dataset = tf.data.TFRecordDataset(
                filename, buffer_size=PREFETCH_DATASET_BUFFER_SIZE,
            num_parallel_reads=16)
            return dataset

        validation_batch_size = val_batch_size
        num_val_steps = ceil(num_val_examples / validation_batch_size / ngpus)

        extract_imagenet_fn = extract_imagenet_fn_mux(image_size=self.image_size)
        extract_imagenet_raw_fn = extract_imagenet_raw_fn_mux(image_size=self.image_size)

        image_size_ = 256

        if tfrecord_path is None:
            train_tfrecord_path = os.path.join("./data/ImageNet/TFRecords/ILSVRC%d/train" % (image_size_))
            val_tfrecord_path = os.path.join("./data/ImageNet/TFRecords/ILSVRC%d/val" % (image_size_))
            train_file_patten = os.path.join(train_tfrecord_path, 'imagenet-train-*')
            val_file_patten = os.path.join(val_tfrecord_path, 'imagenet-val-*')
        else:
            train_tfrecord_path = os.path.join(tfrecord_path, "train_tfrecord")
            val_tfrecord_path = os.path.join(tfrecord_path, "val_tfrecord")
            train_file_patten = os.path.join(train_tfrecord_path, 'train-*')
            val_file_patten = os.path.join(val_tfrecord_path, 'validation-*')

        val_data = tf.data.Dataset.list_files(val_file_patten, shuffle=False)

        val_data = val_data.apply(
            tf.contrib.data.parallel_interleave(
                prefetch_dataset,
                cycle_length=NUM_FILES_FEED,
                sloppy=False))

        val_data = val_data.repeat(-1)

        val_data = val_data.apply(tf.data.experimental.map_and_batch(
            map_func=extract_imagenet_raw_fn,
            drop_remainder=True,
            batch_size=validation_batch_size,
            num_parallel_calls=10))

        val_iterator = val_data.make_one_shot_iterator()

        val_next_step = []
        for i in range(ngpus):
            val_next_step.append(val_iterator.get_next())

        model = Model(self.prototxt, self.solvertxt,
                      data_train_op=val_next_step,
                      data_val_op=val_next_step,
                      inference_type=tf.float32,
                      inference_only=True,
                      ngpus=ngpus)

        model.summary()

        val_top1_accuracy = None
        val_top1_accuracy_summary = tf.Summary()
        val_top1_accuracy_summary.value.add(tag='val_top1_accuracy', simple_value=val_top1_accuracy)

        val_top5_accuracy = None
        val_top5_accuracy_summary = tf.Summary()
        val_top5_accuracy_summary.value.add(tag='val_top5_accuracy', simple_value=val_top5_accuracy)

        val_loss = None
        val_loss_summary = tf.Summary()
        val_loss_summary.value.add(tag='val_loss', simple_value=val_loss)

        best_val_acc = 0
        best_val5_acc = 0

        # Doing validation
        val_loss = 0
        val_acc = 0
        val_top5_acc = 0

        average_latency = 0

        model.eval()

        for j in range(num_val_steps):
            start = time.time()
            loss, acc = model.test_on_batch()
            end = time.time()
            val_loss += loss
            val_acc += acc[0]
            val_top5_acc += acc[1]
            print("[%d/%d], loss=%.5f, acc@1=%.5f, acc@5=%.5f, latency=%.4f ms" %(j+1, num_val_steps,
                                                                       loss, acc[0], acc[1], (end-start)*1000/val_batch_size))
            average_latency += (end-start) * 1000 / num_val_steps / val_batch_size

        val_loss /= num_val_steps
        val_acc /= num_val_steps
        val_top5_acc /= num_val_steps

        print("Val acc@1: %5f, Val acc@5: %.5f, Val loss: %.5f" % (val_acc, val_top5_acc, val_loss))
        print("Average latency: %.4f ms" %average_latency)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val5_acc = val_top5_acc

        model.close()
        return best_val_acc, best_val5_acc, float(model.flops) / 1e6
