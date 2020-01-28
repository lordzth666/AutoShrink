import tensorflow as tf
from api.network.tflayers.layer import Layer
from api.network.Parser import ModelAssign


class Output(Layer):
    def __init__(self, kwargs):
        Layer.__init__(self, kwargs)
        self.name = ModelAssign(kwargs, 'name', None)
        self.output_tensor = None

    def __call__(self, kwargs=None):
        self.output_tensor = ModelAssign(kwargs, 'input')

    def summary(self):
        print("Tensor %s will be an output." %self.output_tensor)


class OutputWithSoftmax(Layer):
    """
    Output a tensor with softmax activation. Usually in the last layer of classification.
    """
    def __init__(self, kwargs):
        Layer.__init__(self, kwargs)
        self.name = ModelAssign(kwargs, 'name', None)
        self.output_tensor = None

    def __call__(self, kwargs=None):
        self.output_tensor = tf.nn.softmax(ModelAssign(kwargs, 'input'))

    def summary(self):
        print("Tensor %s will be an output." %self.output_tensor)
