# Standalone activation functions with proto format.

import tensorflow as tf
from api.network.tflayers.layer import Layer
from api.network.Parser import ModelAssign
from api.network.tflayers.utilities import apply_activation


class Activation(Layer):
    def __init__(self, kwargs):
        """
        Concatenation. Concatenate two tensors if possible.
        :param kwargs: configurations for concatenation layers.
        """
        Layer.__init__(self, kwargs)
        self.name = ModelAssign(kwargs, 'name', None)
        self.activation = ModelAssign(kwargs, 'activation', None)
        self.input_shape = None
        self.output_shape = None
        self.output_tensor = None

        self.num_trainable_parameters = 0
        self.num_non_trainable_parameters = 0
        self.shared_trainable_parameters = 0
        self.mem_cost = 0
        self.MACs = 0
        self.peak_activation_mem = 0

    def __call__(self, kwargs=None):
        """
        Call the activation module and return the output tensor.
        :param kwargs: configurations.
        :return:
        """
        input_tensor = ModelAssign(kwargs, 'input_tensor', None)

        with tf.variable_scope(self.name, tf.AUTO_REUSE) as scope:
            try:
                output = apply_activation(input_tensor, self.activation)
            except Exception as e:
                print(input_tensor)
                raise e
            self.output_shape = output.get_shape()[1:]
            self.MACs = int(self.output_shape[0]) * int(self.output_shape[1]) * int(self.output_shape[2])

        # For activation we do not need to calculate the memory cost since they are usually fused into conv/fc layers.
        self.mem_cost = 0
        self.peak_activation_mem = self.mem_cost
        self.output_tensor = output

    def summary(self):
        """
        Write a summary of the current layer. (e.g. shape change, parameters etc.)
        :return:
        """
        format_str = '|Activation' + ' ' * (17)
        conv_str = "--->%s" %( self.output_shape)
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
        print('%s(%s)' %(self.name, self.activation))
        print("MEM: %.2f M" %(float(self.mem_cost) / 1e6))
        print("PEAK MEM: %.2f K" %(float(self.peak_activation_mem / 1e3)))
