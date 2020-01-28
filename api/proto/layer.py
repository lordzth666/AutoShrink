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

from api.backend import G

def Activation_Proto(name,
                     input,
                     activation='linear'):
    ret = []
    ret.append('[Activation]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('activation=%s' % activation)
    ret.append('')

    return ret


def BatchNorm_Proto(name,
                    input,
                    activation='linear'):
    ret = []
    ret.append('[BatchNorm]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('activation=%s' % activation)
    ret.append('')
    return ret

def Convolutional_Proto(name,
                        input,
                        filters,
                        kernel_size,
                        strides=1,
                        padding='SAME',
                        activation='relu',
                        batchnorm=False,
                        regularizer_strength=G.CONV_DEFAULT_REG,
                        dropout=0.00,
                        use_bias=True,
                        trainable=True):
    ret = []
    ret.append('[Convolutional]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('filters=%d' % filters)
    ret.append('kernel_size=%d' % kernel_size)
    ret.append('strides=%d' % strides)
    ret.append('padding=%s' % padding)
    ret.append('activation=%s' % activation)
    ret.append('regularizer_strength=%e' % regularizer_strength)
    ret.append('dropout=%.3f' %dropout)
    ret.append('use_bias=%s' %use_bias)
    ret.append('trainable=%s' %trainable)

    if batchnorm:
        ret.append('batchnorm=True')
    else:
        ret.append('batchnorm=False')
    ret.append('')
    return ret


def Convolutional1D_Proto(name,
                            input,
                            filters,
                            kernel_size,
                            strides=1,
                            padding='SAME',
                            activation='relu',
                            batchnorm=False,
                            regularizer_strength=G.CONV_DEFAULT_REG,
                            dropout=0.00,
                            hyper=False,
                            hyper_zdims=None,
                            hyper_basic_block_size=None,
                            hyper_hidden=None,
                            use_bias=True,
                            trainable=True):
    ret = []
    ret.append('[Convolutional1D]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('filters=%d' % filters)
    ret.append('kernel_size=%d' % kernel_size)
    ret.append('strides=%d' % strides)
    ret.append('padding=%s' % padding)
    ret.append('activation=%s' % activation)
    ret.append('regularizer_strength=%e' % regularizer_strength)
    ret.append('dropout=%.3f' %dropout)
    ret.append('use_bias=%s' %use_bias)
    ret.append('trainable=%s' %trainable)

    if batchnorm:
        ret.append('batchnorm=True')
    else:
        ret.append('batchnorm=False')
    if hyper:
        ret.append('hyper=True')
        ret.append('hyper_zdims=%d' % hyper_zdims)
        ret.append('hyper_basic_block_size=%s' % hyper_basic_block_size)
        ret.append('hyper_hidden=%d' % hyper_hidden)

    ret.append('')
    return ret


def SeparableConv_Proto(name,
                        input,
                        filters,
                        kernel_size,
                        strides=1,
                        padding='SAME',
                        activation='relu',
                        batchnorm=False,
                        dropout=0.00,
                        regularizer_strength=G.CONV_DEFAULT_REG,
                        use_bias=True,
                        trainable=True):
    ret = []
    ret.append('[SeparableConv]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('filters=%d' % filters)
    ret.append('kernel_size=%d' % kernel_size)
    ret.append('strides=%d' % strides)
    ret.append('padding=%s' % padding)
    ret.append('activation=%s' % activation)
    ret.append('regularizer_strength=%E' % regularizer_strength)
    ret.append('trainable=%s' %trainable)
    ret.append('dropout=%.3f' %dropout)
    if batchnorm:
        ret.append('batchnorm=True')
    else:
        ret.append('batchnorm=False')

    ret.append('use_bias=%s' %use_bias)
    ret.append('')
    return ret


def DepthwiseConv_Proto(name,
                        input,
                        kernel_size,
                        strides=1,
                        padding='SAME',
                        activation='relu',
                        depthwise_multiplier=1,
                        batchnorm=False,
                        dropout=0.00,
                        regularizer_strength=G.CONV_DEFAULT_REG,
                        use_bias=True,
                        trainable=True):
    ret = []
    ret.append('[DepthwiseConv]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('kernel_size=%d' % kernel_size)
    ret.append('depthwise_multiplier=%d' %depthwise_multiplier)
    ret.append('strides=%d' % strides)
    ret.append('padding=%s' % padding)
    ret.append('activation=%s' % activation)
    ret.append('regularizer_strength=%E' % regularizer_strength)
    ret.append('dropout=%.3f' %dropout)
    ret.append('trainable=%s' %trainable)
    if batchnorm:
        ret.append('batchnorm=True')
    else:
        ret.append('batchnorm=False')

    ret.append('use_bias=%s' %use_bias)
    ret.append('')
    return ret


def MaxPool_proto(name,
                  input,
                  pool_size=2,
                  strides=2,
                  padding='SAME'):
    ret = []
    ret.append('[MaxPool]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('pool_size=%d' % pool_size)
    ret.append('strides=%d' % strides)
    ret.append('padding=%s' %padding)
    ret.append('')
    return ret


def MaxPool1D_proto(name,
                  input,
                  pool_size=2,
                  strides=2,
                    padding='SAME'):
    ret = []
    ret.append('[MaxPool1D]')
    ret.append('name=%s' %name)
    ret.append('input=%s' %input)
    ret.append('pool_size=%d' %pool_size)
    ret.append('strides=%d' %strides)
    ret.append('padding=%s' %padding)
    ret.append('')
    return ret


def AvgPool1D_proto(name,
                  input,
                  pool_size=2,
                  strides=2):
    ret = []
    ret.append('[AvgPool1D]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('pool_size=%d' % pool_size)
    ret.append('strides=%d' % strides)
    ret.append('')
    return ret


def AvgPool_proto(name,
                  input,
                  pool_size=2,
                  strides=2):
    ret = []
    ret.append('[AvgPool]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('pool_size=%d' % pool_size)
    ret.append('strides=%d' % strides)
    ret.append('')
    return ret


def GlobalAvgPool_proto(name,
                  input):
    ret = []
    ret.append('[GlobalAvgPool]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('')
    return ret


def GlobalAvgPool1D_proto(name,
                  input):
    ret = []
    ret.append('[GlobalAvgPool1D]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('')
    return ret


def Input_proto(name,
          input_shape,
          dtype='float32',
          mean=-1,
          std=-1):
    ret = []
    ret.append('[Input]')
    ret.append('name=%s' % name)
    ret.append('input_shape=%s' % input_shape)
    ret.append('dtype=%s' % dtype)
    ret.append('mean=%s' %mean)
    ret.append('std=%s' %std)
    ret.append('')
    return ret


def Flatten_proto(name,
                  input):
    ret = []
    ret.append('[Flatten]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('')
    return ret


def Identity_proto(name,
                  input):
    ret = []
    ret.append('[Identity]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('')
    return ret


def Output_proto(name,
                 input):
    ret = []
    ret.append('[Output]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('')
    return ret


def Dropout_proto(name,
                input,
                dropout=0.0):
    ret = []
    ret.append('[Dropout]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('dropout=%s' % dropout)
    ret.append('')
    return ret


def Dense_proto(name,
                input,
                units,
                dropout=0.0,
                activation='linear',
                batchnorm=True,
                regularizer_scale=G.FC_DEFAULT_REG,
                use_bias=True):
    ret = []
    ret.append('[Dense]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('units=%d' % units)
    ret.append('dropout=%s' % dropout)
    ret.append('activation=%s' % activation)
    ret.append('batchnorm=%s' % batchnorm)
    ret.append('regularizer_strength=%e' % regularizer_scale)
    ret.append('use_bias=%s' %use_bias)
    ret.append('')
    return ret


def Concat_proto(name,
                 input):
    ret = []
    ret.append('[Concat]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('')
    return ret

def Add_proto(name,
              input,
              activation):
    ret = []
    ret.append('[Add]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('activation=%s' % activation)
    ret.append('')
    return ret


def Add_n_proto(name,
                input,
                activation='linear'):
    ret = []
    ret.append('[Add_n]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('activation=%s' %activation)
    ret.append('')
    return ret


def Accuracy_proto(name,
                   logits,
                   labels):
    ret = []
    ret.append('[Accuracy]')
    ret.append('name=%s' % name)
    ret.append('logits=%s' % logits)
    ret.append('labels=%s' % labels)
    ret.append('')
    return ret


def Softmax_proto(name,
                  input,
                  labels,
                  label_smoothing=0.0):
    ret = []
    ret.append('[SoftmaxLoss]')
    ret.append('name=%s' % name)
    ret.append('input=%s' % input)
    ret.append('labels=%s' % labels)
    ret.append("label_smoothing=%s" %label_smoothing)
    ret.append('')
    return ret

def TopkAcc_proto(name,
                  logits,
                  labels,
                  k):
    ret = []
    ret.append('[TopkAcc]')
    ret.append('name=%s' % name)
    ret.append('logits=%s' % logits)
    ret.append('labels=%s' % labels)
    ret.append('k=%d' % k)
    ret.append('')
    return ret


