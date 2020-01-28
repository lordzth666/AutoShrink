import tensorflow as tf
from api.network.tflayers.layer import Layer
from api.network.Parser import ModelAssign, Str2List

"""
*****************************
Input Layer
*****************************
Prototxt Prototype
[Input]
name=input
input_shape=[32,32,3]
"""
class Input(Layer):
    def __init__(self, kwargs):
        Layer.__init__(self, kwargs)
        self.name = ModelAssign(kwargs, 'name', None)
        self.input_shape = Str2List(ModelAssign(kwargs, 'input_shape', None))
        self.dtype = ModelAssign(kwargs, 'dtype', 'float32')
        self.output_shape = None
        self.output_tensor = None

        self.mean = ModelAssign(kwargs, 'mean', None)
        self.std = ModelAssign(kwargs, 'std', None)

        self.num_trainable_parameters = 0
        self.num_non_trainable_parameters = 0
        self.shared_trainable_parameters = 0
        self.MACs = 0

    def __call__(self, kwargs=None):
        if kwargs["feed_tensor"] is None:
            tf.logging.info("Using placeholder...")
            if self.input_shape is None:
                output = tf.placeholder(shape=[None], name=self.name, dtype=self.dtype)
            else:
                output = tf.placeholder(shape=[None] + list(self.input_shape), name=self.name, dtype=self.dtype)
        else:
            output = kwargs['feed_tensor']
        if self.mean != "-1" and self.std != "-1" and self.mean is not None and self.std is not None:
            print("Normalizing...")
            _mean = Str2List(self.mean, dtype=float)
            _std = Str2List(self.std, dtype=float)
            output = tf.cast(output, tf.float32)
            output = tf.identity((output-_mean) / _std, name='preprocessed_input')
            output = tf.stop_gradient(output, name='input_finalized_stop_gradient')
        self.output_shape = self.input_shape
        self.output_tensor = output

    def summary(self):
        format_str = '|Input(%s)' %self.name + ' ' * (23-len(self.name))
        conv_str = "%s" %(self.input_shape)
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
