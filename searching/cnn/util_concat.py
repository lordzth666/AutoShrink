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

import tensorflow as tf
import numpy as np

from api.proto.layer import *
from api.backend import G

def add_to_name_scope(
        name,
        scope=None):
    """
    Add name to scope
    :param name:
    :param scope:
    :return:
    """
    if scope is None:
        return name
    else:
        return scope + "/" + name


def make_divisible(filters, divisible_by=4):
    filters = max(round(filters / divisible_by), 1) * divisible_by
    return filters


def get_leaf_nodes(mask):
    """
    Collect the leaf nodes in the mask. Remove activation function for these layers.
    :param mask:
    :return:
    """
    nodes = []
    n = np.shape(mask)[0]
    for i in range(n):
        has_connection = False
        for j in range(n):
            if mask[i][j]:
                has_connection = True
                break
        if not has_connection:
            nodes.append(i)
    return nodes


def convert_mask_to_proto(mask,
                          ops_def,
                          input,
                          scale=1.0,
                          scope=None,
                          strides=1,
                          activation="relu",
                          use_bias=False,
                          batchnorm=True,
                          use_residual=False,
                          bottleneck_factor=4):
    """
    Convert the mask weight matrix to prototxt representation
    :param mask:
    :param ops_def:
    :param input:
    :param scope:
    :return: list of prototxt, final op name
    """
    nodes = np.shape(mask)[0]

    leafs = get_leaf_nodes(mask)

    out_filters = int(ops_def[0]['filters']*scale)
    out_filters = make_divisible(out_filters)
    if not use_residual:
        base_filters = out_filters
    else:
        base_filters = out_filters // bottleneck_factor

    identity_name = add_to_name_scope("node_0", scope)
    # first op is always identity
    if strides == 1:
        ret = Identity_proto(name=identity_name,
                   input=input)
    else:
        ret = MaxPool_proto(name=identity_name,
                            input=input,
                            strides=strides,
                            pool_size=strides)
    concat_cnt = 0
    nodes_queue = [0]
    for i in range(nodes):
        if i in leafs and G.EXEC_CONV_MODE == 'conv-bn-relu' and use_residual:
            activation_ = 'linear'
        else:
            activation_ = activation
        ltype = ops_def[i]['type']
        connected = []
        for j in range(i):
            if not j in nodes_queue:
                continue
            if mask[j][i] == 1:
                node_name = add_to_name_scope("node_%d" %j, scope)
                connected.append(node_name)
                nodes_queue.append(i)
        if len(connected) == 0:
            continue
        # maybe concat or identity
        if len(connected) > 1:
            concat_name = add_to_name_scope("concat_%d" %concat_cnt, scope)
            ret += Concat_proto(name=concat_name,
                             input=connected)
            ifstream = concat_name
            concat_cnt += 1
        else:
            ifstream = connected[0]

        ofstream = add_to_name_scope("node_%d" %i, scope=scope)
        # Insert layer of choice.
        if ltype == "Convolutional":
            # enable stride for only the first op
            ret += Convolutional_Proto(name=ofstream,
                                       input=ifstream,
                                       filters=base_filters,
                                       kernel_size=ops_def[i]['kernel_size'],
                                       strides=1,
                                       padding='SAME',
                                       activation=activation_,
                                       batchnorm=batchnorm,
                                       use_bias=use_bias)
        elif ltype == "SeparableConv":
            ret += SeparableConv_Proto(name=ofstream,
                                       input=ifstream,
                                       filters=base_filters,
                                       kernel_size=ops_def[i]['kernel_size'],
                                       strides=1,
                                       padding='SAME',
                                       activation=activation_,
                                       batchnorm=batchnorm,
                                       use_bias=use_bias)
        elif ltype == "DepthwiseConv":
            ret += DepthwiseConv_Proto(name=ofstream,
                                       input=ifstream,
                                       kernel_size=ops_def[i]['kernel_size'],
                                       strides=1,
                                       depthwise_multiplier=ops_def[i]['depthwise_multiplier'],
                                       padding='SAME',
                                       activation=activation_,
                                       batchnorm=batchnorm,
                                       use_bias=use_bias)
        else:
            print(ltype)
            raise NotImplementedError

    # Inspect leaf nodes and concat
    out_degree = np.zeros(nodes)
    in_degree = np.zeros(nodes)
    in_degree[0] = 1
    for i in range(nodes):
        for j in range(i+1, nodes):
            if mask[i][j] == 1 and i in nodes_queue and j in nodes_queue:
                out_degree[i] += 1
                in_degree[j] += 1

    # Create concat_pool list
    concat_pool_names = []
    for i in range(nodes):
        if out_degree[i] == 0 and in_degree[i] != 0:
            concat_pool_names.append(add_to_name_scope("node_%d" %i, scope))
    # Write Concat Pool list
    concat_name = add_to_name_scope("concat_pool", scope)
    if len(concat_pool_names) != 1:
        ret += Concat_proto(name=concat_name,
                     input=concat_pool_names)
    else:
        ret += Identity_proto(name=concat_name,
                              input=concat_pool_names[0])
    if use_residual:
        out_name = concat_name
        # Then add residual connections together, id mapping is fused into add.
        residual_output_name = add_to_name_scope("residual_out", scope)
        # Add linear bottleneck.
        residual_activation = 'linear'
        ret += Add_proto(residual_output_name,
                         input=[input, out_name],
                         activation=residual_activation)
        output_node_name = residual_output_name
    else:
        output_node_name = concat_name

    return ret, output_node_name


def inspect_num_edges(mask):
    nodes_queue = [0]
    nodes = np.shape(mask)[0]
    edges = 0
    for i in range(nodes):
        if not i in nodes_queue:
            continue
        for j in range(i+1, nodes):
            if mask[i][j] == 1:
                nodes_queue.append(j)
                edges += 1
    return edges
