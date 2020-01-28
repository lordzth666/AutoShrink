import tensorflow as tf
from api.network.tflayers.layer import Layer
from api.network.Parser import ModelAssign
from api.network.tflayers.utilities import convolution2d, apply_activation

from api.backend import G

"""
Prototxt Prototype
[Add]
name=added
input=conv1,conv2
axis=-1
"""


class Add(Layer):
    """
    Elementwise adding of two tensors.
    Considering adding identity mapping if the last dimension does not match. By convention,
    the first tensor should be input as output dimension will be mapped to the second tensor.
    """
    def __init__(self, kwargs):
        Layer.__init__(self, kwargs)
        self.name = ModelAssign(kwargs, 'name', None)
        self.axis = ModelAssign(kwargs, 'axis', -1)
        self.activation = ModelAssign(kwargs, 'activation', None)
        self.input_shape = None
        self.output_shape = None
        self.output_tensor = None

        self.num_trainable_parameters = 0
        self.num_non_trainable_parameters = 0
        self.shared_trainable_parameters = 0
        self.MACs = 0

        self.peak_activation_mem = 0

    def __call__(self, kwargs=None):
        input_tensor = ModelAssign(kwargs, 'input_tensor', None)
        is_training = ModelAssign(kwargs, 'is_training')
        if not isinstance(input_tensor, list):
            assert NotImplementedError, "The input tensor list must have exactly two tensors"
            raise NotImplementedError
        op1 = input_tensor[0]
        op2 = input_tensor[1]

        if G.data_format == 'channels_last':
            _filter_len1 = int(op1.get_shape()[-1])
            _filter_len2 = int(op2.get_shape()[-1])
            # try to guess strides
            op1_feat_size = int(op1.get_shape()[1])
            op2_feat_size = int(op2.get_shape()[1])
        else:
            _filter_len1 = int(op1.get_shape()[1])
            _filter_len2 = int(op2.get_shape()[1])
            # try to guess strides
            op1_feat_size = int(op1.get_shape()[-1])
            op2_feat_size = int(op2.get_shape()[-1])

        if op1_feat_size % op2_feat_size != 0:
            tf.logging.warning("op1_feat_size must be divided by op2_feat_size, but %d vs %d. Please double check" %(op1_feat_size, op2_feat_size))
        strides = op1_feat_size // op2_feat_size
        tf.logging.info("Inferred Strides=%d for op1_feat_size=%d vs op2_feat_size=%d" %(strides, op1_feat_size, op2_feat_size))
        if _filter_len1 ==_filter_len2 and strides == 1:
            tf.logging.info(strides)
            with tf.variable_scope(self.name, tf.AUTO_REUSE) as scope:
                output = op1 + op2
                if G.EXEC_CONV_MODE == 'conv-bn-relu':
                    output = apply_activation(output, self.activation)

                self.mem_cost = self.num_non_trainable_parameters + self.num_trainable_parameters
                self.input_shape = op1.get_shape()
                self.output_shape = op2.get_shape()
                self.peak_activation_mem += int(self.input_shape[1]) * int(self.input_shape[2]) * int(
                    self.input_shape[3])
                self.peak_activation_mem += int(self.output_shape[1]) * int(self.output_shape[2]) * int(
                    self.output_shape[3])

        else:
            with tf.variable_scope(self.name, tf.AUTO_REUSE) as scope:
                if G.EXEC_CONV_MODE == 'relu-conv-bn':
                    activation_ = self.activation
                    print("Activation is conducted at conv...")
                else:
                    activation_ = 'linear'
                    print("Activation is conducted at add...")

                print("Intializing Conv Layer with Default L2-reg.")
                id_mapping = convolution2d(op1,
                                           filters=_filter_len2,
                                           kernel_size=1,
                                           padding='same',
                                           activation=activation_,
                                           batchnorm=True,
                                           strides=strides,
                                           use_bias=False,
                                           is_training=is_training,
                                           mode=G.EXEC_CONV_MODE)
                output = id_mapping + op2
                if G.EXEC_CONV_MODE == 'conv-bn-relu':
                    output = apply_activation(output, self.activation)
                self.mem_cost = self.num_non_trainable_parameters + self.num_trainable_parameters
                self.input_shape = id_mapping.get_shape()
                self.output_shape = op2.get_shape()
                self.peak_activation_mem += int(self.input_shape[1]) * int(self.input_shape[2]) * int(
                    self.input_shape[3])
                self.peak_activation_mem += int(self.output_shape[1]) * int(self.output_shape[2]) * int(
                    self.output_shape[3])

            self.num_trainable_parameters = _filter_len2 * _filter_len1
            self.num_non_trainable_parameters = 2 * _filter_len2
            self.MACs = op2_feat_size * op2_feat_size * _filter_len1 * _filter_len2
            self.out_shape = output.get_shape()

        self.output_tensor = output

    def summary(self):
        format_str = '|Add' + ' ' * (19)
        conv_str = "--->%s" %( self.output_shape)
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
        ts = '%s'%None
        tstr = '|      ' + ts + ' ' * (14-len(ts)) + '|'
        format_str += tstr
        print(format_str)
        print(self.name)
        print("FLOPS(MAC): %f M" % (float(self.MACs) / 1000000))
        print("Peak MEM: %.2f K" %(float(self.peak_activation_mem) / 1e3))



"""
Prototxt Prototype
[Add_n]
name=added
input=conv1,conv2,conv3
axis=-1
"""

#FIXME: maintain it for channel_first operation.
class Add_n(Layer):
    """
    Elementwise add of N tensors. The n tensors have to be within the same shape.
    """

    def __init__(self, kwargs):
        Layer.__init__(self, kwargs)
        self.name = ModelAssign(kwargs, 'name', None)
        self.axis = ModelAssign(kwargs, 'axis', -1)
        self.activation = ModelAssign(kwargs, 'activation', None)
        self.input_shape = None
        self.output_shape = None
        self.output_tensor = None

        self.num_trainable_parameters = 0
        self.num_non_trainable_parameters = 0
        self.shared_trainable_parameters = 0
        self.MACs = 0


    def __call__(self, kwargs=None):
        input_tensor = ModelAssign(kwargs, 'input_tensor', None)
        is_training = ModelAssign(kwargs, 'is_training')
        if not isinstance(input_tensor, list):
            assert NotImplementedError, "The input tensor list must have exactly two tensors"
            raise NotImplementedError
        # Add trainable parameter to control scale
        base_filters = int(input_tensor[-1].get_shape()[-1])
        input_tensor_scaled = []
        with tf.variable_scope(self.name) as scope:
            id = 0
            for tensor in input_tensor:
                scale = tf.get_variable(name='scale_%d' %id, dtype=tf.float32,
                                        shape=(),
                                        initializer=tf.initializers.ones())
                current_filter = int(tensor.get_shape()[-1])
                if G.EXEC_CONV_MODE == 'relu-conv-bn':
                    activation_ = self.activation
                else:
                    activation_ = 'linear'
                if current_filter != base_filters:
                    tensor = convolution2d(tensor,
                                           filters=base_filters,
                                           kernel_size=1,
                                           padding='same',
                                           activation=activation_,
                                           batchnorm=True,
                                           strides=1,
                                           use_bias=False,
                                           is_training=is_training,
                                           mode=G.EXEC_CONV_MODE)
                    print(tensor)
                print(scale)
                scaled_tensor = tensor * scale
                input_tensor_scaled.append(scaled_tensor)
                id += 1

            output = tf.add_n(input_tensor_scaled, name='added_tensor')
            if G.EXEC_CONV_MODE == 'conv-bn-relu':
                output = apply_activation(output, self.activation)

            self.output_tensor = output
        self.output_shape = self.output_tensor.get_shape()[1:]

    def summary(self):
        format_str = '|Add_n' + ' ' * (19)
        conv_str = "--->%s" %( self.output_shape)
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
        ts = '%s'%None
        tstr = '|      ' + ts + ' ' * (14-len(ts)) + '|'
        format_str += tstr
        print(format_str)
        print(self.name)
        print("FLOPS(MAC): %f M" % (float(self.MACs) / 1000000))

    def _get_common_filters(self, filter_list):
        pass