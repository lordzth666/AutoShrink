import tensorflow as tf
from api.network.tflayers.layer import Layer
import api.network.tflayers.utilities as util
from api.network.Parser import ModelAssign, Str2List
from api.backend import G

"""
*****************************
Convolution Layer
*****************************
Prototxt Prototype
[Convolutional]
name=conv_pre1
input=input
filters=32
kernel_size=3
strides=2
padding=SAME
activation=elu
regularizer_strength=0.000100
batchnorm=True
"""


class Convolutional(Layer):
    def __init__(self, kwargs):
        Layer.__init__(self, kwargs)
        self.name = ModelAssign(kwargs, 'name', None)
        self.input = ModelAssign(kwargs, 'input', None)
        self.filters = ModelAssign(kwargs, 'filters', 32)
        self.kernel_size = ModelAssign(kwargs, 'kernel_size', 3)
        self.strides = ModelAssign(kwargs, 'strides', 1)
        self.padding = ModelAssign(kwargs, 'padding', 'SAME')
        self.hyper = ModelAssign(kwargs, 'hyper', False)
        self.batchnorm = ModelAssign(kwargs, 'batchnorm', False)
        self.activation = ModelAssign(kwargs, 'activation', 'relu')
        self.dropout = ModelAssign(kwargs, 'dropout', 0.0)
        self.dropout_tensor = None
        self.use_bias = ModelAssign(kwargs, 'use_bias', True)
        self.num_trainable_parameters = 0
        self.num_non_trainable_parameters = 0
        self.shared_trainable_parameters = 0
        self.mem_cost = 0
        self.peak_activation_mem = 0
        self.skip_from_names = None
        self.MACs = 0
        # HyperNetwork parameters
        if not self.hyper:
            self.zdims = None
            self.layer_info = None
            self.basic_block_size = None
            self.hidden = None
        else:
            self.zdims = ModelAssign(kwargs, 'hyper_zdims', 4)
            self.layer_info = tf.placeholder(dtype=tf.float32, shape=[1, self.zdims], name=self.name+'layer_info')
            self.basic_block_size = Str2List(ModelAssign(kwargs, 'hyper_basic_block_size', None))
            self.hidden = ModelAssign(kwargs, 'hyper_hidden', 16)

        self.input_shape = None
        self.output_shape = None

        self.output_tensor = None

    def __call__(self, kwargs=None):
        input_tensor = ModelAssign(kwargs, 'input_tensor', None)
        self.input_shape = input_tensor.get_shape()[1:]
        self.skip_from_names = ModelAssign(kwargs, 'skip_from_names', None)
        input_filters = int(self.input_shape[-1])
        skip_from = ModelAssign(kwargs, 'skip_from', None)
        initializer = ModelAssign(kwargs, 'initializer', G.BACKEND_DEFAULT_CONV_INITIALIZER)
        regularizer = ModelAssign(kwargs, 'regularizer', G.BACKEND_DEFAULT_REGULARIZER)
        self.is_training = ModelAssign(kwargs, 'is_training')
        regularizer_strength = ModelAssign(kwargs, 'regularizer_strength', 1e-4)
        # infer filters during compiling
        if self.filters == 1:
            self.filters = int(input_tensor.get_shape()[-1])
        print("Intializing Conv Layer with L2-reg=%f" %regularizer_strength)

        with tf.variable_scope(self.name) as scope:
            if not self.hyper:
                output = util.convolution2d(inputs=input_tensor,
                                            kernel_size=self.kernel_size,
                                            filters=self.filters,
                                            strides=self.strides,
                                            padding=self.padding,
                                            batchnorm=self.batchnorm,
                                            activation=self.activation,
                                            initializer=initializer(),
                                            regularizer=regularizer(regularizer_strength),
                                            use_bias=self.use_bias,
                                            is_training=self.is_training,
                                            mode=G.EXEC_CONV_MODE
                                            )
                self.output_shape = output.get_shape()[1:]
                # Normal
                self.num_trainable_parameters += (self.kernel_size * self.kernel_size * input_filters) * self.filters
                if self.use_bias:
                    self.num_trainable_parameters += self.filters
                # FLOPS-MAC
                self.MACs += int(self.input_shape[0]) * int(self.input_shape[1]) * \
                             self.kernel_size * self.kernel_size * input_filters * self.filters / self.strides / self.strides
                if self.batchnorm and not G.DISABLE_BATCHNORM_COUNT:
                    self.num_trainable_parameters += 2 * self.filters
                    self.num_non_trainable_parameters += 2 * self.filters
                    self.MACs += int(self.output_shape[0]) * int(self.output_shape[1]) * int(self.output_shape[2]) * 2
            else:
                raise NotImplementedError

            self.mem_cost = self.num_non_trainable_parameters + self.num_trainable_parameters
            self.peak_activation_mem += int(self.input_shape[0]) * int(self.input_shape[1]) * int(self.input_shape[2])
            self.peak_activation_mem += int(self.output_shape[0]) * int(self.output_shape[1]) * int(self.output_shape[2])

            if self.dropout > 1e-12:
                self.dropout_tensor = tf.placeholder(dtype=tf.float32)
                output = tf.nn.dropout(output, keep_prob=1-self.dropout_tensor, name="dropout")

            self.output_tensor = output

            ret_dict = {}
            if self.hyper:
                ret_dict.update({'layerinfo': {self.layer_info:
                                                   [self.kernel_size, self.kernel_size, input_filters, self.filters]}})
            if self.dropout > 1e-12:
                ret_dict.update({'dropout': self.dropout_tensor})
            return ret_dict

    def summary(self):
        if not self.hyper:
            format_str = "|Convolutional(%d,%d)" %(self.kernel_size, self.kernel_size)
            format_str += ' ' * (31-len(format_str))
        else:
            format_str = "|HyperConv(%s)(%d,%d)" %(self.name, self.kernel_size, self.kernel_size)
            format_str += ' ' * (31-len(format_str))
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
        print(self.name)
        print("FLOPS(MAC): %f M" % (float(self.MACs) / 1000000))
        print("MEM: %.2f M" %(float(self.mem_cost) / 1e6))
        print("Peak MEM: %.2f K" %(float(self.peak_activation_mem) / 1e3))
        pass


"""
*****************************
Depthwise Convolution Layer
*****************************
Prototxt Prototype
[DepthwiseConv]
name=conv_pre1
input=input
filters=32
kernel_size=3
strides=2
padding=SAME
activation=elu
regularizer_strength=0.000100
batchnorm=True
"""
class DepthwiseConv(Layer):
    def __init__(self, kwargs):
        Layer.__init__(self, kwargs)
        self.name = ModelAssign(kwargs, 'name', None)
        self.input = ModelAssign(kwargs, 'input', None)
        self.kernel_size = ModelAssign(kwargs, 'kernel_size', 3)
        self.strides = ModelAssign(kwargs, 'strides', 1)
        self.padding = ModelAssign(kwargs, 'padding', 'SAME')
        self.hyper = ModelAssign(kwargs, 'hyper', False)
        self.batchnorm = ModelAssign(kwargs, 'batchnorm', False)
        self.activation = ModelAssign(kwargs, 'activation', 'relu')
        self.use_bias = ModelAssign(kwargs, 'use_bias', True)
        self.depthwise_multiplier=ModelAssign(kwargs, 'depthwise_multiplier', 1)

        self.num_trainable_parameters = 0
        self.num_non_trainable_parameters = 0
        self.shared_trainable_parameters = 0
        self.mem_cost = 0
        self.skip_from_names = None
        self.MACs = 0
        self.peak_activation_mem = 0

        # HyperNetwork parameters
        if not self.hyper:
            self.zdims = None
            self.layer_info = None
            self.basic_block_size = None
            self.hidden = None
        else:
            self.zdims = ModelAssign(kwargs, 'hyper_zdims', 4)
            self.layer_info = tf.placeholder(dtype=tf.float32, shape=[1, self.zdims], name=self.name+'layer_info')
            self.basic_block_size = Str2List(ModelAssign(kwargs, 'hyper_basic_block_size', None))
            self.hidden = ModelAssign(kwargs, 'hyper_hidden', 16)

        self.input_shape = None
        self.output_shape = None

        self.output_tensor = None

    def __call__(self, kwargs=None):
        input_tensor = ModelAssign(kwargs, 'input_tensor', None)
        self.input_shape = input_tensor.get_shape()[1:]
        self.skip_from_names = ModelAssign(kwargs, 'skip_from_names', None)
        skip_from = ModelAssign(kwargs, 'skip_from', None)
        self.is_training = ModelAssign(kwargs, 'is_training')
        initializer = ModelAssign(kwargs, 'initializer', G.BACKEND_DEFAULT_CONV_INITIALIZER)
        regularizer = ModelAssign(kwargs, 'regularizer', G.BACKEND_DEFAULT_REGULARIZER)
        regularizer_strength = ModelAssign(kwargs, 'regularizer_strength', 1e-4)

        # infer filters during compiling
        self.filters = int(input_tensor.get_shape()[-1]) * self.depthwise_multiplier

        with tf.variable_scope(self.name) as scope:
            self.output_tensor = util.depthwise_conv2d(inputs=input_tensor,
                                           depth_multiplier=self.depthwise_multiplier,
                                           kernel_size=self.kernel_size,
                                           strides=self.strides,
                                           padding=self.padding,
                                           batchnorm=self.batchnorm,
                                           activation=self.activation,
                                           initializer=initializer(),
                                           regularizer=regularizer(regularizer_strength),
                                           use_bias=self.use_bias,
                                           is_training=self.is_training,
                                           mode=G.EXEC_CONV_MODE
                                        )

            self.output_shape = self.output_tensor.get_shape()[1:]
            # Normal
            self.num_trainable_parameters += (self.kernel_size * self.kernel_size) * self.filters
            if self.use_bias:
                self.num_trainable_parameters += self.filters
            self.MACs += self.kernel_size * self.kernel_size * self.filters * int(self.output_shape[0]) * int(self.output_shape[1])
            if self.batchnorm and not G.DISABLE_BATCHNORM_COUNT:
                self.num_trainable_parameters += 2 * self.filters
                self.num_non_trainable_parameters += 2 * self.filters
                self.MACs += int(self.output_shape[0]) * int(self.output_shape[1]) * int(self.output_shape[2]) * 2

            self.mem_cost = self.num_non_trainable_parameters + self.num_trainable_parameters
            self.peak_activation_mem += int(self.input_shape[0]) * int(self.input_shape[1]) * int(self.input_shape[2])
            self.peak_activation_mem += int(self.output_shape[0]) * int(self.output_shape[1]) * int(self.output_shape[2])


    def summary(self):
        format_str = "|DepthwiseConv(%d,%d)" %(self.kernel_size, self.kernel_size)
        format_str += ' ' * (31-len(format_str))
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
        print(self.name)
        print("FLOPS(MAC): %.2f M" % (float(self.MACs) / 1000000))
        print("MEM: %.2f M" %(float(self.mem_cost) / 1e6))
        print("Peak MEM: %.2f K" %(float(self.peak_activation_mem) / 1e3))

        pass


class SeparableConv(Layer):
    def __init__(self, kwargs):
        Layer.__init__(self, kwargs)
        self.name = ModelAssign(kwargs, 'name', None)
        self.input = ModelAssign(kwargs, 'input', None)
        self.filters = ModelAssign(kwargs, 'filters', 32)
        self.kernel_size = ModelAssign(kwargs, 'kernel_size', 3)
        self.strides = ModelAssign(kwargs, 'strides', 1)
        self.padding = ModelAssign(kwargs, 'padding', 'SAME')
        self.hyper = ModelAssign(kwargs, 'hyper', False)
        self.batchnorm = ModelAssign(kwargs, 'batchnorm', False)
        self.activation = ModelAssign(kwargs, 'activation', 'relu')
        self.use_bias = ModelAssign(kwargs, 'use_bias', True)
        self.num_trainable_parameters = 0
        self.num_non_trainable_parameters = 0
        self.shared_trainable_parameters = 0
        self.skip_from_names = None
        self.MACs = 0
        self.peak_activation_mem = 0

        self.dropout = ModelAssign(kwargs, 'dropout', 0.0)
        self.dropout_tensor = None
        # HyperNetwork parameters
        if not self.hyper:
            self.zdims = None
            self.layer_info = None
            self.basic_block_size = None
            self.hidden = None
        else:
            raise NotImplementedError

        self.input_shape = None
        self.output_shape = None

        self.output_tensor = None

    def __call__(self, kwargs=None):
        input_tensor = ModelAssign(kwargs, 'input_tensor', None)
        self.input_shape = input_tensor.get_shape()[1:]
        self.skip_from_names = ModelAssign(kwargs, 'skip_from_names', None)
        skip_from = ModelAssign(kwargs, 'skip_from', None)
        self.is_training = ModelAssign(kwargs, 'is_training')
        initializer = ModelAssign(kwargs, 'initializer', G.BACKEND_DEFAULT_CONV_INITIALIZER)
        regularizer = ModelAssign(kwargs, 'regularizer', G.BACKEND_DEFAULT_REGULARIZER)
        regularizer_strength = ModelAssign(kwargs, 'regularizer_strength', 1e-4)
        input_filter = int(self.input_shape[-1])
        # infer filters during compiling
        if self.filters == 1:
            self.filters = int(input_tensor.get_shape()[-1])
        print("Intializing Separable Conv Layer with L2-reg=%f" %regularizer_strength)

        with tf.variable_scope(self.name) as scope:
            output = util.separable_conv2d(inputs=input_tensor,
                                           kernel_size=self.kernel_size,
                                           filters=self.filters,
                                           strides=self.strides,
                                           padding=self.padding,
                                           batchnorm=self.batchnorm,
                                           depthwise_activation=self.activation,
                                           pointwise_activation=self.activation,
                                           initializer=initializer(),
                                           regularizer=regularizer(regularizer_strength),
                                           use_bias=self.use_bias,
                                           is_training=self.is_training,
                                           mode=G.EXEC_CONV_MODE)
            self.output_shape = output.get_shape()[1:]
            # +depthwise
            self.num_trainable_parameters += (self.kernel_size * self.kernel_size) * input_filter
            if self.use_bias:
                self.num_trainable_parameters += input_filter
            self.MACs += self.kernel_size * self.kernel_size * input_filter * \
                         int(self.input_shape[0]) * int(self.input_shape[1])

            # +pointwise
            self.num_trainable_parameters += (input_filter) * self.filters
            if self.use_bias:
                self.num_trainable_parameters += self.filters
            self.MACs += input_filter * self.filters * int(self.output_shape[0]) * int(self.output_shape[1])
            if self.batchnorm and not G.DISABLE_BATCHNORM_COUNT:
                self.num_trainable_parameters += 2 * (input_filter + self.filters)
                self.num_non_trainable_parameters += 2 * (input_filter + self.filters)
                self.MACs += 2 * int(self.output_shape[0]) * int(self.output_shape[1]) * int(self.output_shape[2]) * 2

            self.mem_cost = self.num_non_trainable_parameters + self.num_trainable_parameters
            self.peak_activation_mem += int(self.input_shape[0]) * int(self.input_shape[1]) * int(self.input_shape[2])
            self.peak_activation_mem += int(self.output_shape[0]) * int(self.output_shape[1]) * int(self.output_shape[2])

            if self.dropout > 1e-12:
                self.dropout_tensor = tf.placeholder(dtype=tf.float32)
                output = tf.nn.dropout(output, keep_prob=1-self.dropout_tensor, name="dropout")

            self.output_tensor = output
            ret_dict = {}

            if self.dropout > 1e-12:
                ret_dict.update({'dropout': self.dropout_tensor})
            return ret_dict

    def summary(self):
        format_str = "|SeparableConv(%d,%d)" %(self.kernel_size, self.kernel_size)
        format_str += ' ' * (31-len(format_str))
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
        print(self.name)
        print("FLOPS(MAC): %.2f M" % (float(self.MACs) / 1000000))
        print("MEM: %.2f M" %(float(self.mem_cost) / 1e6))
        print("Peak MEM: %.2f K" %(float(self.peak_activation_mem) / 1e3))

        pass


class DilatedConv(Layer):
    def __init__(self, kwargs):
        Layer.__init__(self, kwargs)
        pass

    def __call__(self, kwargs=None):
        pass

    def summary(self):
        pass
