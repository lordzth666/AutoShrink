from api.network.tflayers.layer import Layer
from api.network.Parser import ModelAssign
from api.network.tflayers.utilities import batch_normalization

import tensorflow as tf


class BatchNorm(Layer):
    def __init__(self, kwargs):
        Layer.__init__(self, kwargs)
        self.name = ModelAssign(kwargs, 'name', None)
        self.axis = ModelAssign(kwargs, 'axis', -1)
        self.trainable = ModelAssign(kwargs, 'trainable', True)
        self.input_shape = None
        self.output_shape = None
        self.output_tensor = None

        self.num_trainable_parameters = 0
        self.num_non_trainable_parameters = 0
        self.shared_trainable_parameters = 0
        self.MACs = 0

    def __call__(self, kwargs=None):
        input_tensor = ModelAssign(kwargs, 'input_tensor', None)
        self.input_shape = input_tensor.get_shape()[1:]
        self.is_training = ModelAssign(kwargs, 'is_training')
        with tf.variable_scope(self.name, tf.AUTO_REUSE) as scope:
            output = batch_normalization(input_tensor, is_training=self.is_training,
                                         activation='linear', trainable=self.trainable)
        self.output_shape = output.get_shape()[1:]
        self.output_tensor = output
        self.MACs = int(self.output_shape[0]) * int(self.output_shape[1]) * int(self.output_shape[2]) * 2

    def summary(self):
        format_str = 'BatchNorm' + ' ' * (22-len(self.name))
        conv_str = "%s-->%s" %(self.input_shape, self.output_shape)
        space = " " * (36-len(conv_str))
        format_str += "|" + conv_str + space
        ts = '%s'%0
        tstr = '|      ' + ts + ' ' * (22-len(ts))
        format_str += tstr
        ts = '%s'%0
        tstr = '|      ' + ts + ' ' * (26-len(ts))
        format_str += tstr
        ts = '%s'%0
        tstr = '|      ' + ts + ' ' * (15-len(ts))
        format_str += tstr
        ts = '%s'%None
        tstr = '|      ' + ts + ' ' * (14-len(ts)) + '|'
        format_str += tstr
        print(format_str)
        print(self.name)
