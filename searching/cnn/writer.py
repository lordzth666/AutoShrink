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

from api.proto.lazy_loader import *
from api.backend import G

def make_divisible(n, divided_by=4):
    return divided_by * round(n / divided_by)

class ProtoWriter:
    def __init__(self, name=None):
        self.proto_list = []
        self.name = name
        self.ifstream = None

    def add_header(self, task='cifar10'):
        if task in headers[G.PROTO_BACKEND].keys():
            proto, self.ifstream = headers[G.PROTO_BACKEND][task](self.name)
            self.add(proto)

    def add(self, proto):
        self.proto_list.append(proto)

    def finalized(self, task='cifar10', outstream_name=None):
        if task in finals[G.PROTO_BACKEND].keys():
            proto = finals[G.PROTO_BACKEND][task](outstream_name)
            self.add(proto)
        else:
            raise NotImplementedError
        pass

    def scaling(self, factor):
        for id in range(len(self.proto_list)):
            for item_id in range(len(self.proto_list[id])):
                if len(self.proto_list[id][item_id])<7:
                    continue
                if self.proto_list[id][item_id][:7] == 'filters':
                    fnum = int(self.proto_list[id][item_id].split('=')[1])
                    self.proto_list[id][item_id] = 'filters=%d' %make_divisible(fnum*factor)
                if self.proto_list[id][item_id][:12] == 'base_filters':
                    fnum = int(self.proto_list[id][item_id].split('=')[1])
                    self.proto_list[id][item_id] = 'base_filters=%d' %make_divisible(fnum*factor)
                if self.proto_list[id][item_id][:11] == 'out_filters':
                    fnum = int(self.proto_list[id][item_id].split('=')[1])
                    self.proto_list[id][item_id] = 'out_filters=%d' %make_divisible(fnum*factor)

    def dump(self, fp):
        for out in self.proto_list:
            for item in out:
                fp.write(item + "\n")

        pass

    def _set_global_properties(self, property, value, exclude_property=[], count=-1):
        """
        Set global property of one specific value
        :param property:
        :param value:
        :param count: maximum counts for setting.
        :return: None
        """
        property_length = len(property)
        cnt = 0
        for id in range(len(self.proto_list)):
            if cnt == count:
                break
            for item_id in range(len(self.proto_list[id])):
                if len(self.proto_list[id][item_id])<property_length:
                    continue
                if self.proto_list[id][item_id][:property_length] == property:
                    rvalue = self.proto_list[id][item_id].split("=")[1]
                    if rvalue in exclude_property:
                        continue
                    self.proto_list[id][item_id] = '%s=%s' %(property, value)
                    cnt += 1
        pass

    def set_global_regularization(self, reg_value):
        """
        :param reg_value:
        :return:
        """
        self._set_global_properties("regularizer_strength", reg_value)

    def set_global_dropout(self, dropout_value):
        """
        :param dropout_value:
        :return:
        """
        self._set_global_properties('dropout', dropout_value)

    def set_bias_flag(self, flag):
        """
        :param flag:
        :return:
        """
        self._set_global_properties('use_bias', flag)

    def set_batchnorm_flag(self, flag):
        """
        Setting the batchnorm flag.
        :param flag: True/False
        :return:
        """
        self._set_global_properties("batchnorm", flag)

    def set_activation(self, activation):
        """
        Set the activation function of each layer in proto.
        :param activation: activation name.
        :return:
        """
        self._set_global_properties('activation', activation, exclude_property=['linear', 'softmax'])
        pass

    def convert_conv2d_to_separable(self):
        """
        Converting all of the conv2d op to separable conv2d
        :return: None
        """
        self._convert_layer("Convolutional", "SeparableConv")
        pass

    def convert_separable_to_conv2d(self):
        """
        Converting all of the separable conv2d op to conv2d op
        :return:
        """
        self._convert_layer("SeparableConv", "Convolutional")
        pass

    def _convert_layer(self, old_layer_name, new_layer_name):
        for id in range(len(self.proto_list)):
            for item_id in range(len(self.proto_list[id])):
                if self.proto_list[id][item_id] == "":
                    continue
                if self.proto_list[id][item_id][0] == '[':
                    parse_str = self.proto_list[id][item_id][1:-1]
                    if parse_str == old_layer_name:
                        self.proto_list[id][item_id] = '[%s]' %new_layer_name

    def freeze_first_n_layers(self, n):
        """
        Freeze first n layers during training.
        :param n:
        :return:
        """
        self._set_global_properties('trainable', False, count=n)
        pass
