import tensorflow as tf
from api.network.tflayers.layer import Layer
from api.network.Parser import ModelAssign

"""
Prototxt Prototype

[Accuracy]
name=accuracy
logits=flatten
labels=label
"""


class Accuracy(Layer):
    def __init__(self, kwargs):
        """
        Accuracy.
        :param kwargs:
        """
        Layer.__init__(self, kwargs)
        self.name = ModelAssign(kwargs, 'name', None)
        self.logits = ModelAssign(kwargs, 'logits', None)
        self.labels = ModelAssign(kwargs, 'labels', None)
        self.output_tensor = None

    def __call__(self, kwargs=None, method='training'):
        """
        Call the accuracy class and return an metric tensor.
        :param kwargs: configurations
        :param method: scope.
        :return:
        """
        logits = ModelAssign(kwargs, 'logits', None)
        labels = ModelAssign(kwargs, 'labels', None)
        self.output_tensor = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1),
                                               tf.argmax(labels, 1)), dtype=tf.float32))

    def summary(self):
        """
        Write a summary of the current layer. (e.g. shape change, parameters etc.)
        :return:
        """
        return


class TopkAcc(Layer):
    def __init__(self, kwargs):
        """
        Top-k accuracy for large-scale image classification. (ImageNet)
        :param kwargs: Configurations
        """
        Layer.__init__(self, kwargs)
        self.name = ModelAssign(kwargs, 'name', None)
        self.logits = ModelAssign(kwargs, 'logits', None)
        self.labels = ModelAssign(kwargs, 'labels', None)
        self.k = ModelAssign(kwargs, 'k', 5)
        self.output_tensor = None

    def __call__(self, kwargs=None, method='training'):
        logits = ModelAssign(kwargs, 'logits', None)
        labels = ModelAssign(kwargs, 'labels', None)
        with tf.device("/cpu:0"):
            accuracy = tf.keras.metrics.top_k_categorical_accuracy(labels, logits, k=self.k)
        self.output_tensor = accuracy

    def summary(self):
        return
