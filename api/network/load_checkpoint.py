from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
import tensorflow as tf


def grep_gpu_attr(name):
    base_name, gpu_id = name.split(":")
    return base_name


def get_vars_to_restore(ckpt_file):
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_file)
    name_collections = []
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in sorted(var_to_shape_map):
        name_collections.append(key)
    return name_collections


def get_tensors_by_name(tensor_names):
    total_varlist = tf.all_variables()
    tensor_collection = []
    for var in total_varlist:
        name_to_restore = grep_gpu_attr(var.name)
        if name_to_restore in tensor_names:
            tensor_collection.append(var)
    return tensor_collection
