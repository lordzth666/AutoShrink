import tensorflow as tf
from api.network.tflayers.layer import Layer
from api.network.Parser import ModelAssign
from api.network.tflayers.utilities import concat

class Concat(Layer):
    def __init__(self, kwargs):
        """
        Concatenation. Concatenate two tensors if possible.
        :param kwargs: configurations for concatenation layers.
        """
        Layer.__init__(self, kwargs)
        self.name = ModelAssign(kwargs, 'name', None)

        self.axis = ModelAssign(kwargs, 'axis', -1)
        self.input_shape = None
        self.output_shape = None
        self.output_tensor = None

        self.num_trainable_parameters = 0
        self.num_non_trainable_parameters = 0
        self.shared_trainable_parameters = 0
        self.mem_cost = 0
        self.MACs = 0
        self.peak_activation_mem = 0


    def __call__(self, kwargs=None, renorm=False):
        """
        Call the concat module and return the output tensor.
        :param kwargs: configurations.
        :param renorm: Renorm activations for concatenation. (experimental feature)
        :return:
        """
        input_tensor = ModelAssign(kwargs, 'input_tensor', None)

        if renorm:
            renormed_inputs = []
            # Calculate l2-norm sum
            l2_norm = []
            for t in input_tensor:
                l2_norm.append(tf.nn.l2_loss(t))
            l2_norm = tf.reduce_mean(l2_norm)
            for t in input_tensor:
                renormed_inputs.append(tf.nn.l2_normalize(t) * l2_norm)
            input_tensor = renormed_inputs

        with tf.variable_scope(self.name, tf.AUTO_REUSE) as scope:
            try:
                output = concat(input_tensor, axis=self.axis)
            except Exception as e:
                print(input_tensor)
                raise e
        try:
            self.output_shape = output.get_shape()[1:]
        except Exception as e:
            print(self.name)
            raise e
        # Calculate Mem Cost
        if not isinstance(input_tensor, list):
            t = input_tensor
            tshape = t.get_shape()[1:]
            if len(tshape) == 3:
                self.mem_cost += int(tshape[0]) * int(tshape[1]) * int(tshape[2])
            elif len(tshape) == 2:
                self.mem_cost += int(tshape[0]) * int(tshape[1])
        else:
            for t in input_tensor:
                tshape = t.get_shape()[1:]
                if len(tshape) == 3:
                    self.mem_cost += int(tshape[0]) * int(tshape[1]) * int(tshape[2])
                elif len(tshape) == 2:
                    self.mem_cost += int(tshape[0]) * int(tshape[1])
        self.peak_activation_mem = self.mem_cost
        self.output_tensor = output

    def summary(self):
        """
        Write a summary of the current layer. (e.g. shape change, parameters etc.)
        :return:
        """
        format_str = '|Concat' + ' ' * (22)
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
        print(self.name)
        print("MEM: %.2f M" %(float(self.mem_cost) / 1e6))
        print("PEAK MEM: %.2f K" %(float(self.peak_activation_mem / 1e3)))
