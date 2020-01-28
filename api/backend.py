"""
Copyright (c) <2019> <CEI Lab, Duke University>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os


try:
    import numpy as np
    import tensorflow as tf
except Exception:
    pass

class _Backend:
    """
    This is the backend for this NAS framework. Random seeds/System settings are configured here.
    It also helps configure some hyperparameters such as Batch Normalization epsilon/momentum,
    weight initializer etc.
    """
    def __init__(self, BACKEND="tensorflow", PROTO_BACKEND="native"):
        pass

        # Set random seed.
        tf.set_random_seed(233)
        np.random.seed(233)
        #-----------BACKEND Settings--------
        # Can change that to other APIs in the future.
        self.BACKEND = BACKEND

        # disable batchnorm count for fair comparison.
        self.DISABLE_BATCHNORM_COUNT = True

        # Time seed
        self.TIME_SEED = 20190311

        # Enable mix precision.
        self.data_format = "channels_last"
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

        if self.BACKEND == "tensorflow":
            self.BACKEND_DEFAULT_INITIALIZER = lambda: tf.initializers.glorot_normal()
            self.BACKEND_DEFAULT_CONV_INITIALIZER = self.BACKEND_DEFAULT_INITIALIZER
            self.BACKEND_DEFAULT_FC_INITIALIZER = self.BACKEND_DEFAULT_INITIALIZER
            self.BACKEND_DEFAULT_REGULARIZER = tf.keras.regularizers.l2
            tf.logging.set_verbosity(tf.logging.INFO)
        else:
            raise NotImplementedError
        #---------DEFAULT REG Settings---------
        # Specify default regularization setting
        self.FC_DEFAULT_REG = 1e-15
        self.CONV_DEFAULT_REG = 1e-15
        self.SEPCONV_DEFAULT_REG = 1e-15

        # Default setting. This is the recommended setting for CIFAR-10 dataset.
        self.BN_EPSILON = 1e-5
        self.BN_MOMENTUM = 0.9

        # Specify no regularizatino value
        self.NO_REG = 1e-20

        # Specify conv order.
        self.EXEC_CONV_MODE = 'conv-bn-relu'

        # PROTO BACKEND
        self.PROTO_BACKEND = PROTO_BACKEND
        if self.PROTO_BACKEND == "native":
            pass
        else:
            raise NotImplementedError
        # Use all gpu FLAG.
        self.USE_ALL_GPU = 0

    def set_backend(self, BACKEND, seed=233):
        self.BACKEND = BACKEND
        if self.BACKEND == "tensorflow":
            import tensorflow as tf
            self.BACKEND_DEFAULT_INITIALIZER = lambda: tf.initializers.variance_scaling(seed=seed, scale=2.0, mode='fan_out')
            self.BACKEND_DEFAULT_CONV_INITIALIZER = self.BACKEND_DEFAULT_INITIALIZER
            self.BACKEND_DEFAULT_FC_INITIALIZER = lambda: tf.initializers.random_normal(seed=seed, stddev=1e-3)
            self.BACKEND_DEFAULT_REGULARIZER = tf.contrib.layers.l2_regularizer
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        else:
            print("Unknown backend: %s"  %BACKEND)
            raise NotImplementedError

    def set_proto_backend(self, PROTO_BACKEND):
        # PROTO BACKEND
        self.PROTO_BACKEND = PROTO_BACKEND
        if self.PROTO_BACKEND == "native":
            pass
        elif self.PROTO_BACKEND == "caffe":
            pass
        else:
            print("Unknown proto backend: %s"  %PROTO_BACKEND)
            raise NotImplementedError

    def config_bn_params(self, epsilon=1e-5, momentum=0.9):
        """
        Configure Batch Normalization parameters and Momentum.
        :param epsilon: 'epsilon' for BN layers.
        :param momentum: 'momentum' for BN layers. It is recommended to use 0.9, 0.99, 0.999 etc. as
        a good start.
        :return: None
        """
        self.BN_EPSILON = epsilon
        self.BN_MOMENTUM = momentum

    def set_initializer(self, initializer, seed=233):
        """
        Configure the weight initialiation method.
        :param initializer: Initializer type. Can be 'xavier', 'he-normal', 'zeros' and 'default'.
        :param seed: Random seed.
        :return: None
        """
        if initializer == 'xavier':
            self.BACKEND_DEFAULT_INITIALIZER = lambda: tf.initializers.glorot_normal(seed=seed)
            self.BACKEND_DEFAULT_CONV_INITIALIZER = self.BACKEND_DEFAULT_INITIALIZER
            self.BACKEND_DEFAULT_FC_INITIALIZER = self.BACKEND_DEFAULT_INITIALIZER

        elif initializer == 'he-normal':
            self.BACKEND_DEFAULT_INITIALIZER = lambda: tf.initializers.he_normal(seed=seed)
            self.BACKEND_DEFAULT_CONV_INITIALIZER = self.BACKEND_DEFAULT_INITIALIZER
            self.BACKEND_DEFAULT_FC_INITIALIZER = self.BACKEND_DEFAULT_INITIALIZER
        elif initializer == "default":
            self.BACKEND_DEFAULT_INITIALIZER = lambda: tf.initializers.variance_scaling(seed=seed, scale=2.0, mode='fan_out')
            self.BACKEND_DEFAULT_CONV_INITIALIZER = self.BACKEND_DEFAULT_INITIALIZER
            self.BACKEND_DEFAULT_FC_INITIALIZER = lambda: tf.initializers.random_normal(seed=seed, stddev=0.01)
        elif initializer == 'zeros':
            self.BACKEND_DEFAULT_INITIALIZER = lambda: tf.initializers.zeros()
            self.BACKEND_DEFAULT_CONV_INITIALIZER = self.BACKEND_DEFAULT_INITIALIZER
            self.BACKEND_DEFAULT_FC_INITIALIZER = self.BACKEND_DEFAULT_INITIALIZER

    def set_conv_triplet(self, triplet):
        """
        Configure conv-bn order. This method is deprecated.
        :param triplet: Can be 'conv-bn-relu'.
        :return: None
        """
        assert triplet == 'conv-bn-relu'
        self.EXEC_CONV_MODE = triplet
        print("Using %s triplet ..." %triplet)

    def use_all_gpu(self):
        """
        Call this to occupy the full GPU. This is especially helpful during NAS search process.
        :return: None
        """
        self.USE_ALL_GPU = 1

# Instantialize a backend instance.
G = _Backend()
