import tensorflow as tf
from api.network.tflayers.layer import Layer
from api.network.Parser import ModelAssign
import api.network.tflayers.utilities as util
from api.backend import *


class Dense(Layer):
    def __init__(self, kwargs):
        Layer.__init__(self, kwargs)
        self.name = ModelAssign(kwargs, 'name', None)
        self.input = ModelAssign(kwargs, 'input', None)
        self.units = ModelAssign(kwargs, 'units', 10)
        self.use_bias = ModelAssign(kwargs, 'use_bias', True)
        self.batchnorm = ModelAssign(kwargs, 'batchnorm', False)
        self.trainable = ModelAssign(kwargs, 'trainable', True)
        self.activation = ModelAssign(kwargs, 'activation', 'relu')
        self.dropout = ModelAssign(kwargs, 'dropout', 0.0)
        self.skip_from_names = None
        self.input_shape = None
        self.output_shape = None
        self.output_tensor = None
        self.dropout_tensor = None

        # Params
        self.num_trainable_parameters = 0
        self.num_non_trainable_parameters = 0
        self.shared_trainable_parameters = 0
        self.MACs = 0
        self.mem_cost = 0
        self.peak_activation_mem = 0

    def __call__(self, kwargs=None):
        skip_from = ModelAssign(kwargs, 'skip_from', None)
        self.skip_from_names = ModelAssign(kwargs, 'skip_from_names', None)
        input_tensor = ModelAssign(kwargs, 'input_tensor', None)
        self.input_shape = input_tensor.get_shape()[1:]
        self.is_training = ModelAssign(kwargs, 'is_training', None)
        print(self.is_training)
        in_dim = int(input_tensor.get_shape()[-1])
        initializer = ModelAssign(kwargs, 'initializer', G.BACKEND_DEFAULT_FC_INITIALIZER)
        regularizer = ModelAssign(kwargs, 'regularizer', G.BACKEND_DEFAULT_REGULARIZER)
        regularizer_strength = ModelAssign(kwargs, 'regularizer_strength', 1e-4)
        print("Intializing Dense Layer with L2-reg=%f" %regularizer_strength)
        with tf.variable_scope(self.name) as scope:
            output = util.dense(input_tensor,
                                units=self.units,
                                activation=self.activation,
                                batchnorm=self.batchnorm,
                                initializer=initializer(),
                                regularizer=regularizer(regularizer_strength),
                                is_training=self.is_training,
                                trainable=self.trainable,
                                use_bias=self.use_bias
                                )
            self.num_trainable_parameters += (in_dim+1) * self.units
            self.MACs += int(in_dim) * self.units
            if self.batchnorm:
                self.num_non_trainable_parameters += 2 * self.units
                self.num_trainable_parameters += 2 * self.units
                self.MACs += int(self.units) * 2

            if self.dropout > 1e-12:
                self.dropout_tensor = tf.placeholder(dtype=tf.float32, shape=())
                output = tf.nn.dropout(output, keep_prob=1-self.dropout_tensor, name="dropout")
            self.output_tensor = output
            self.output_shape = self.output_tensor.get_shape()[1:]
            self.mem_cost = self.num_non_trainable_parameters + self.num_trainable_parameters
            self.peak_activation_mem += int(self.input_shape[0])
            self.peak_activation_mem += int(self.output_shape[0])

            if self.dropout > 1e-12:
                return {'dropout': self.dropout_tensor}

    def summary(self):
        format_str = '|Dense(%s)   ' %self.name + ' ' * (20-len(self.name))
        conv_str = "%s-->%s" %(self.input_shape, self.output_shape)
        space = " " * (36-len(conv_str))
        format_str += "|" + conv_str + space
        ts = '%s'%self.num_trainable_parameters
        tstr = '|      ' + ts + ' ' * (22-len(ts))
        format_str += tstr
        ts = '%s'%self.num_non_trainable_parameters
        tstr = '|      ' + ts + ' ' * (26-len(ts))
        format_str += tstr
        ts = '%s'%self.shared_trainable_parameters
        tstr = '|      ' + ts + ' ' * (15-len(ts))
        format_str += tstr
        ts = '%s'%self.skip_from_names
        tstr = '|      ' + ts + ' ' * (14-len(ts)) + '|'
        format_str += tstr
        print(format_str)
        print("FLOPS(MAC): %.2f M" % (float(self.MACs) / 1e6))
        print("MEM: %.2f M" %(float(self.mem_cost) / 1e6))
        print("Peak MEM: %.2f K" %(float(self.peak_activation_mem) / 1e3))

