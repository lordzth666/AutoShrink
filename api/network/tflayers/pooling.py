import tensorflow as tf
from api.network.tflayers.layer import Layer
import api.network.tflayers.utilities as util
from api.network.Parser import ModelAssign


class MaxPool(Layer):
    def __init__(self, kwargs):
        Layer.__init__(self, kwargs)
        self.strides = ModelAssign(kwargs, 'strides', 2)
        self.pool_size = ModelAssign(kwargs, 'pool_size', 2)
        self.padding = ModelAssign(kwargs, 'padding', 'SAME')
        self.name = ModelAssign(kwargs, 'name', None)

        self.num_trainable_parameters = 0
        self.num_non_trainable_parameters = 0
        self.shared_trainable_parameters = 0
        self.mem_cost = 0
        self.MACs = 0
        self.peak_activation_mem = 0

        self.input_shape = None
        self.output_shape = None

        self.output_tensor = None

    def __call__(self, kwargs=None):
        input_tensor = ModelAssign(kwargs, 'input_tensor', None)
        self.input_shape = input_tensor.get_shape()[1:]
        with tf.variable_scope(self.name, tf.AUTO_REUSE) as scope:
            self.output_tensor = util.maxpool(inputs=input_tensor,
                                strides=self.strides,
                                ksize=self.pool_size,
                                padding=self.padding)
        self.output_shape = self.output_tensor.get_shape()[1:]
        self.mem_cost += int(self.input_shape[0]) * int(self.input_shape[1]) * int(self.input_shape[2])
        self.peak_activation_mem += int(self.input_shape[0]) * int(self.input_shape[1]) * int(self.input_shape[2])
        self.peak_activation_mem += int(self.output_shape[0]) * int(self.output_shape[1]) * int(self.output_shape[2])

    def summary(self):
        format_str = '|MaxPool(%s)   ' %self.name + ' ' * (18-len(self.name))
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
        print("MEM: %.2f M" %(float(self.mem_cost) / 1e6))
        print("Peak MEM: %.2f M" %(float(self.peak_activation_mem) / 1e6))



class AvgPool(Layer):
    def __init__(self, kwargs):
        Layer.__init__(self, kwargs)
        self.strides = ModelAssign(kwargs, 'strides', 2)
        self.pool_size = ModelAssign(kwargs, 'pool_size', 2)
        self.padding = ModelAssign(kwargs, 'padding', 'SAME')
        self.name = ModelAssign(kwargs, 'name', None)

        self.num_trainable_parameters = 0
        self.num_non_trainable_parameters = 0
        self.shared_trainable_parameters = 0
        self.MACs = 0
        self.mem_cost = 0
        self.peak_activation_mem = 0

        self.input_shape = None
        self.output_shape = None

        self.output_tensor = None

    def __call__(self, kwargs=None):
        input_tensor = ModelAssign(kwargs, 'input_tensor', None)
        self.input_shape = input_tensor.get_shape()[1:]
        with tf.variable_scope(self.name, tf.AUTO_REUSE) as scope:
            output = util.avgpool(inputs=input_tensor,
                                  strides=self.strides,
                                  ksize=self.pool_size,
                                  padding=self.padding)
        self.output_shape = output.get_shape()[1:]
        self.mem_cost += int(self.input_shape[0]) * int(self.input_shape[1]) * int(self.input_shape[2])
        self.output_tensor = output
        self.peak_activation_mem += int(self.input_shape[0]) * int(self.input_shape[1]) * int(self.input_shape[2])
        self.peak_activation_mem += int(self.output_shape[0]) * int(self.output_shape[1]) * int(self.output_shape[2])

    def summary(self):
        format_str = '|AvgPool(%s)   ' %self.name + ' ' * (18-len(self.name))
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
        print("MEM: %.2f M" %(float(self.mem_cost) / 1e6))
        print("Peak MEM: %.2f M" %(float(self.peak_activation_mem) / 1e6))


class GlobalAvgPool(Layer):
    def __init__(self, kwargs):
        Layer.__init__(self, kwargs)
        self.padding = ModelAssign(kwargs, 'padding', 'SAME')
        self.name = ModelAssign(kwargs, 'name', None)

        self.num_trainable_parameters = 0
        self.num_non_trainable_parameters = 0
        self.shared_trainable_parameters = 0
        self.MACs = 0
        self.mem_cost = 0
        self.peak_activation_mem = 0

        self.input_shape = None
        self.output_shape = None

        self.output_tensor = None

    def __call__(self, kwargs=None):
        input_tensor = ModelAssign(kwargs, 'input_tensor', None)
        self.input_shape = input_tensor.get_shape()[1:]
        with tf.variable_scope(self.name, tf.AUTO_REUSE) as scope:
            output = util.globalAvgPool(inputs=input_tensor)
        self.output_shape = output.get_shape()[1:]
        self.output_tensor = output
        self.mem_cost += int(self.input_shape[0]) * int(self.input_shape[1]) * int(self.input_shape[2])
        self.peak_activation_mem += int(self.input_shape[0]) * int(self.input_shape[1]) * int(self.input_shape[2])
        self.peak_activation_mem += int(self.output_shape[0]) * int(self.output_shape[1]) * int(self.output_shape[2])

    def summary(self):
        format_str = '|GlobalAvgPool(%s)   ' %self.name + ' ' * (12-len(self.name))
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
        print("MEM: %.2f M" %(float(self.mem_cost) / 1e6))
        print("Peak MEM: %.2f M" %(float(self.peak_activation_mem) / 1e6))

