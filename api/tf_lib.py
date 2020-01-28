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

from api.network.tflayers.activation import *
from api.network.tflayers.convolutional import *
from api.network.tflayers.flatten import *
from api.network.tflayers.input import *
from api.network.tflayers.identity import *
from api.network.tflayers.dense import *
from api.network.tflayers.dropout import *
from api.network.tflayers.pooling import *
from api.network.tflayers.losses import *
from api.network.tflayers.accuracy import *
from api.network.tflayers.concat import *
from api.network.tflayers.output import *
from api.network.decay.decay import *
from api.network.tflayers.add import *
from api.network.tflayers.batchnorm import *

import tensorflow as tf


def sgd(lr):
    return tf.train.GradientDescentOptimizer(learning_rate=lr)

def sgd_momentum(lr, momentum=.9):
    return tf.train.MomentumOptimizer(learning_rate=lr, momentum=momentum, use_nesterov=False)


def sgd_nesterov(lr, momentum=.9):
    return tf.train.MomentumOptimizer(learning_rate=lr, momentum=momentum, use_nesterov=True)


def rmsprop_imagenet(lr, momentum=.9):
    return tf.train.RMSPropOptimizer(learning_rate=lr, momentum=momentum, epsilon=1.0)


def rmsprop_cifar10(lr, momentum=.9):
    return tf.train.RMSPropOptimizer(learning_rate=lr, momentum=momentum, epsilon=1e-4)


def asgd(lr, momentum=0.9):
    return ASGD(learning_rate=lr)


def adam_imagenet(lr):
    return tf.train.AdamOptimizer(learning_rate=lr, epsilon=1.0, beta1=0.5, beta2=0.999)


def nadam_imagenet(lr):
    return tf.contrib.opt.NadamOptimizer(learning_rate=lr, epsilon=1.0)


def nadam_cifar(lr):
    return tf.contrib.opt.NadamOptimizer(learning_rate=lr, epsilon=1e-4)


LayerObj = {'Activation': Activation,
            'BatchNorm': BatchNorm,
            'Convolutional': Convolutional,
            'DepthwiseConv': DepthwiseConv,
            'SeparableConv': SeparableConv,
            'Dropout': Dropout,
            'Flatten': Flatten,
            'Input': Input,
            'Identity': Identity,
            'Dense': Dense,
            'MaxPool': MaxPool,
            'AvgPool': AvgPool,
            'Add': Add,
            'Add_n': Add_n,
            'GlobalAvgPool': GlobalAvgPool,
            'SoftmaxLoss': SoftmaxLoss,
            'Accuracy': Accuracy,
            'Concat': Concat,
            'YoloLoss': YoloLoss,
            'Output': Output,
            'OutputWithSoftmax': OutputWithSoftmax,
            'TopkAcc': TopkAcc}

Optimizer = {'adam': tf.train.AdamOptimizer,
             'rmsprop': tf.train.RMSPropOptimizer,
             'sgd': tf.train.GradientDescentOptimizer,
             'sgd_momentum': sgd_momentum,
             'sgd_nesterov': sgd_nesterov,
             'rmsprop_imagenet': rmsprop_imagenet,
             'adam_imagenet': adam_imagenet,
             'rmsprop_cifar10': rmsprop_cifar10,
             'nadam_imagenet': nadam_imagenet,
             'nadam_cifar': nadam_cifar,
             'asgd': asgd}

Decay = {'step_decay': step_decay,
         'stepwise_decay': stepwise_decay,
         'cosine_decay': cosine_decay,
         'no_decay': no_decay}

# Define types
inputs_ = ['Input', 'Input_Definite']
layers_ = ['Activation', 'BatchNorm','Convolutional', 'DepthwiseConv', 'SeparableConv', 'Add',
           'Add_n', 'Flatten', 'Dropout', 'Dense', 'MaxPool',
           'AvgPool','GlobalAvgPool', 'Concat', 'Identity', 'GlobalAvgPool1D']
losses_ = ['SoftmaxLoss', 'YoloLoss']
metrics_ = {'Accuracy', 'TopkAcc'}
outputs_ = {'Output', 'OutputWithSoftmax'}
