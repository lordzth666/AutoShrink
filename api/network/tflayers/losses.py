import tensorflow as tf
from api.network.tflayers.layer import Layer
import api.network.tflayers.utilities as util
from api.network.Parser import ModelAssign


class SoftmaxLoss(Layer):
    def __init__(self, kwargs):
        Layer.__init__(self, kwargs)
        self.input_shape = None
        self.name = ModelAssign(kwargs, 'name', None)
        self.label_smoothing = ModelAssign(kwargs, 'label_smoothing', 0.0)
        self.output_tensor = None

        self.num_trainable_parameters = 0
        self.num_non_trainable_parameters = 0
        self.shared_trainable_parameters = 0
        self.MACs = 0

    def __call__(self, kwargs=None, mode="training"):
        self.label = ModelAssign(kwargs, 'label', None)
        self.prediction = ModelAssign(kwargs, 'input', None)
        self.input_shape = self.label.get_shape()[1:]
        self.output_tensor = tf.losses.softmax_cross_entropy(self.label, self.prediction,
                                                             label_smoothing=self.label_smoothing)

    def summary(self):
        format_str = '|SoftmaxLoss(%s)' %self.name + ' ' * (17-len(self.name))
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

