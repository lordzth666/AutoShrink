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

from api.proto.layer import *

def Cifar10_header(model_name='cifar10-model'):
    ret = []
    ret.append('[Model]')
    ret.append('name=%s' % model_name)
    ret.append('pretrain=./models/%s' %model_name)
    #ret.append('pretrain=None')
    ret.append('')
    ret += Input_proto(input_shape=[32, 32, 3], name='input')
    ret += Input_proto(input_shape=[10], name='label')
    ret += Convolutional_Proto(name='conv_pre1',
                               kernel_size=3,
                               strides=1,
                               padding='SAME',
                               filters=32,
                               input='input',
                               activation='relu',
                               batchnorm=True,
                               regularizer_strength=1e-4,
                               use_bias=False)

    return ret, "conv_pre1"


def Cifar10_final(outstream_name=None):
    ret = []
    ret += GlobalAvgPool_proto(name='avg_1k',
                               input=outstream_name)

    ret += Flatten_proto(name='flatten',
                         input='avg_1k')

    ret += Dense_proto(name='logits',
                       input='flatten',
                       units=10,
                       activation='linear',
                       batchnorm=False,
                       use_bias=True,
                       regularizer_scale=1e-5)

    ret += Softmax_proto(name='softmax_loss',
                         input='logits',
                         labels='label')

    ret += Accuracy_proto(name='accuracy',
                          logits='logits',
                          labels='label')
    return ret


def ImageNet_header(model_name='imagenet-model', size=224,
                    use_depthwise=False):
    ret = []

    ret.append('[Model]')
    ret.append('name=%s' %model_name)
    ret.append('pretrain=./models/%s' %model_name)
    ret.append('')
    ret += Input_proto(input_shape=[size, size, 3],
                       name='input')
    ret += Input_proto(input_shape=[1001], name='label')
    ret += Convolutional_Proto(name='conv_pre1',
                               kernel_size=3,
                               strides=2,
                               padding='SAME',
                               filters=32,
                               input='input',
                               activation='relu',
                               batchnorm=True,
                               regularizer_strength=1e-5,
                               use_bias=False,
                               trainable=True)
    out = "conv_pre1"
    return ret, out


def ImageNet_final(outstream_name=None, size=224):
    ret = []

    ret += Convolutional_Proto(name='conv1x1',
                               input=outstream_name,
                               filters=1280,
                               regularizer_strength=1e-6,
                               activation='relu',
                               kernel_size=1,
                               batchnorm=True,
                               use_bias=False)

    ret += GlobalAvgPool_proto(name='avg_1k',
                               input='conv1x1')

    ret += Flatten_proto(name='flatten',
                         input='avg_1k')

    ret += Dropout_proto(name='fc_drop',
                         input='flatten',
                         dropout=0.00)

    ret += Dense_proto(name='logits',
                       input='fc_drop',
                       units=1001,
                       regularizer_scale=1e-6,
                       activation='linear',
                       batchnorm=False)

    ret += Softmax_proto(name='softmax_loss',
                         input='logits',
                         labels='label',
                         label_smoothing=0.1)

    ret += Accuracy_proto(name='accuracy',
                          logits='logits',
                          labels='label')

    ret += TopkAcc_proto(name='top_5_acc',
                         logits='logits',
                         labels='label',
                         k=5)
    return ret


