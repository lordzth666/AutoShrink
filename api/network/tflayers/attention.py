import tensorflow as tf
from api.network.tflayers.layer import Layer
import api.network.tflayers.utilities as util
from api.network.Parser import ModelAssign


"""
Dummy attention layer
"""
class Attention(Layer):
    def __init__(self, kwargs):
        """
        Initialization of the Attention Model Class
        :param kwargs: arguments for configuration
        """
        super(Attention).__init__(kwargs)
        self.method = ModelAssign(kwargs, 'method', 'softmax')
        self.name = ModelAssign(kwargs, 'name', None)
        self.input = ModelAssign(kwargs, 'input', None)
        self.num_heads = ModelAssign(kwargs, 'num_heads', 1)
        self.dropout_rate = ModelAssign(kwargs, 'dropout_rate', 0.00)
        self.casuality = ModelAssign(kwargs, 'casuality', False)
        self.outputs = None
        pass

    def __call__(self, kwargs=None):
        """
        Call the attention module and returns an output tensor
        :param kwargs: configurations
        :return: an output tensor after feeding the input into attention model
        """
        input_tensor = ModelAssign(kwargs, 'input_tensor', None)
        self.input_shape = input_tensor.get_shape()[1:]
        if self.method == 'softmax':
            with tf.variable_scope(self.name, tf.AUTO_REUSE):
                self.outputs = util.softmax_self_attention(inputs=input_tensor)
        else:
            raise NotImplementedError
        pass

    def summary(self):
        """
        Write a summary of the current layer. (e.g. shape change, parameters etc.)
        :return:
        """
        pass



"""
***************************************
MultiHeadAttention Layer for Transformer
***************************************
Prototxt Prototype:

[Attention]
name=attention
queries=decode
keys=memory
values=memory
num_heads=5
dropout_rate=0.00
casuality=False
method='vanilla'
"""
class MultiHeadAttention(Layer):
    def __init__(self, kwargs):
        """
        Initialization of the Attention Model Class
        :param kwargs: arguments for configuration
        """
        super(Attention).__init__(kwargs)
        self.method = ModelAssign(kwargs, 'method', 'softmax')
        self.name = ModelAssign(kwargs, 'name', None)
        self.input = ModelAssign(kwargs, 'input', None)
        self.num_heads = ModelAssign(kwargs, 'num_heads', 1)
        self.dropout_rate = ModelAssign(kwargs, 'dropout_rate', 0.00)
        self.casuality = ModelAssign(kwargs, 'casuality', False)
        self.outputs = None
        pass

    def __call__(self, kwargs=None):
        """
        Call the attention module and returns an output tensor
        :param kwargs: configurations
        :return: an output tensor after feeding the input into attention model
        """
        queries = ModelAssign(kwargs, 'queries', None)
        keys = ModelAssign(kwargs, 'keys', None)
        values = ModelAssign(kwargs, 'values', None)

        if self.method == 'softmax':
            with tf.variable_scope(self.name, tf.AUTO_REUSE):
                self.outputs = util.multihead_attention(queries=queries,
                                                        keys=keys,
                                                        values=values,
                                                        num_heads=self.num_heads,
                                                        dropout_rate=self.dropout_rate,
                                                        causality=self.casuality)
        else:
            raise NotImplementedError
        pass

    def summary(self):
        """
        Write a summary of the current layer. (e.g. shape change, parameters etc.)
        :return:
        """
        pass

