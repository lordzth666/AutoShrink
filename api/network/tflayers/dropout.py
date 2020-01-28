import tensorflow as tf

from api.network.tflayers.layer import Layer
import api.network.tflayers.utilities as util
from api.network.Parser import ModelAssign


class Dropout(Layer):
    def __init__(self, kwargs):
        Layer.__init__(self, kwargs)
        self.name = ModelAssign(kwargs, 'name', None)
        self.dropout = ModelAssign(kwargs, 'dropout', 0.0)
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
        self.dropout_tensor = tf.placeholder(dtype=tf.float32)
        if self.dropout > 1e-12:
            output = tf.nn.dropout(input_tensor, keep_prob=1 - self.dropout_tensor, name="dropout")
        else:
            output = input_tensor
        self.output_shape = output.get_shape()[1:]
        self.output_tensor = output

        if self.dropout > 1e-12:
            return {'dropout': self.dropout_tensor}

    def summary(self):
        format_str = '|Dropout(%s)' %self.name + ' ' * (21-len(self.name))
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
