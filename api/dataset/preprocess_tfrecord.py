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
from api.dataset.random_erasing import random_erasing
from api.backend import G
from tensorflow.python.ops import control_flow_ops


def make_divisible(size, divisible_by=16):
    new_size = round(size / divisible_by) * divisible_by
    return new_size

def distort_aspect_ratio(image,
                         bbox=None,
                         min_object_covered=None,
                         aspect_ratio_range=(0.75, 1.33),
                         area_range=(0.25, 1.0),
                         max_attempts=100,
                         scope=None):
  """Generates cropped_image using a one of the bboxes randomly distorted.
  See `tf.image.sample_distorted_bounding_box` for more documentation.
  Args:
    image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
      image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
      area of the image must contain at least this fraction of any bounding box
      supplied.
    aspect_ratio_range: An optional list of `floats`. The cropped area of the
      image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `floats`. The cropped area of the image
      must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
      region of the image of the specified constraints. After `max_attempts`
      failures, return the entire image.
    scope: Optional scope for name_scope.
  Returns:
    A tuple, a 3-D Tensor cropped_image and the distorted bbox
  """
  if bbox is None:
      bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                         dtype=tf.float32,
                         shape=[1, 1, 4])
  with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].

    # A large fraction of image datasets contain a human-annotated bounding
    # box delineating the region of the image containing the object of interest.
    # We choose to create a new bounding box for the object which is a randomly
    # distorted version of the human-annotated bounding box that obeys an
    # allowed range of aspect ratios, sizes and overlap with the human-annotated
    # bounding box. If no box is supplied, then we assume the bounding box is
    # the entire image.
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    return cropped_image

def apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].
  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.
  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0, fast_mode=False, scope=None):
  """Distort the color of a Tensor image.
  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.
  Args:
    image: 3-D Tensor containing single image in [0, 1].
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=0.4)
        image = tf.image.random_saturation(image, lower=0.6, upper=1.4)
      else:
        image = tf.image.random_saturation(image, lower=0.6, upper=1.4)
        image = tf.image.random_brightness(image, max_delta=0.4)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=0.4)
        image = tf.image.random_saturation(image, lower=0.6, upper=1.4)
        image = tf.image.random_hue(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.6, upper=1.4)
        image = tf.image.random_brightness(image, max_delta=0.4)
        image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
        image = tf.image.random_hue(image, max_delta=0.1)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
        image = tf.image.random_hue(image, max_delta=0.1)
        image = tf.image.random_brightness(image, max_delta=0.4)
        image = tf.image.random_saturation(image, lower=0.6, upper=1.4)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.1)
        image = tf.image.random_saturation(image, lower=0.6, upper=1.4)
        image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
        image = tf.image.random_brightness(image, max_delta=0.4)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)



def _extract_imagenet_raw_fn(tfrecord, image_size):
    features = {
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/class/label': tf.FixedLenFeature([], tf.int64),
        'image/height':  tf.FixedLenFeature([], tf.int64),
        'image/width':  tf.FixedLenFeature([], tf.int64)
    }
    sample = tf.parse_single_example(tfrecord, features)
    image = tf.image.decode_jpeg(sample['image/encoded'], channels=3,
                                 fancy_upscaling=False,
                                 dct_method="INTEGER_FAST")

    image = tf.image.central_crop(image, 0.875)

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    #image = tf.to_float(image)
    #image = tf.multiply(image, 1. / 255)

    # Resize Bicubic
    image = tf.image.resize_images(image, (image_size, image_size), 2)

    image = tf.subtract(image, [0.5,0.5,0.5])
    image = tf.multiply(image, [2.0,2.0,2.0])

    if G.data_format == 'channels_first':
        image = tf.transpose(image, (2, 0, 1))

    label = tf.one_hot(sample['image/class/label'], depth=1001)
    return [image, label]


def _extract_imagenet_fn(tfrecord, image_size, fast_mode=False):
    features = {
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/class/label': tf.FixedLenFeature([], tf.int64),
        'image/height':  tf.FixedLenFeature([], tf.int64),
        'image/width':  tf.FixedLenFeature([], tf.int64)
    }
    sample = tf.parse_single_example(tfrecord, features)
    image = tf.image.decode_jpeg(sample['image/encoded'], channels=3,
                                 fancy_upscaling=False,
                                 dct_method="INTEGER_FAST")

    #image = tf.to_float(image)
    #image = tf.multiply(image, 1. / 255)

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Distort & Crop. `0.75` is well tuned.
    image = distort_aspect_ratio(image, area_range=(0.75, 1))

    num_resize_cases = 4 if fast_mode else 4
    image = apply_with_random_selector(
        image,
        lambda x, method: tf.image.resize_images(x, [image_size, image_size], method),
        num_cases=num_resize_cases)
    # image = tf.image.random_crop(image, [image_size, image_size, 3])
    # Flip
    image = tf.image.random_flip_left_right(image)
    if image_size > 96:
        # distort color
        num_distort_cases = 2 if fast_mode else 4
        image = apply_with_random_selector(
            image,
            lambda x, ordering: distort_color(x, ordering, fast_mode),
            num_cases=num_distort_cases)

    # Random erasing
    image = random_erasing(image)

    image = tf.subtract(image, [0.5,0.5,0.5])
    image = tf.multiply(image, [2.0,2.0,2.0])

    if G.data_format == 'channels_first':
        image = tf.transpose(image, (2, 0, 1))

    label = tf.one_hot(sample['image/class/label'], depth=1001)
    return [image, label]


_supported_imagenet_sizes = [32, 96, 128, 160, 192, 224]

def extract_imagenet_fn_mux(image_size):
    if image_size in _supported_imagenet_sizes:
        return lambda tfrecord: _extract_imagenet_fn(tfrecord, image_size)
    else:
        raise NotImplementedError


def extract_imagenet_raw_fn_mux(image_size):
    if image_size in _supported_imagenet_sizes:
        return lambda tfrecord: _extract_imagenet_raw_fn(tfrecord, image_size)
    else:
        raise NotImplementedError


_cifar_mean = np.asarray((0.491, 0.482, 0.446))
_cifar_std = np.asarray((0.247, 0.243, 0.262))

def extract_cifar10_fn(tfrecord):
    features = {
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    }
    sample = tf.parse_single_example(tfrecord, features)
    image = tf.decode_raw(sample['image_raw'], tf.uint8)
    image = tf.reshape(image, [32, 32, 3])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
    image = tf.image.random_crop(image, (32, 32, 3))
    image = tf.image.random_flip_left_right(image)
    #image = random_erasing(image)
    # Normalize
    image = tf.subtract(image, _cifar_mean)
    image = tf.divide(image, _cifar_std)
    label = tf.one_hot(sample['label'], depth=10)

    if G.data_format == 'channels_first':
        image = tf.transpose(image, (2, 0, 1))

    return image, label


def extract_cifar10_proxy_fn(tfrecord):
    features = {
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    }
    sample = tf.parse_single_example(tfrecord, features)
    image = tf.decode_raw(sample['image_raw'], tf.uint8)
    image = tf.reshape(image, [32, 32, 3])
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Normalize
    image = tf.subtract(image, _cifar_mean)
    image = tf.divide(image, _cifar_std)
    label = tf.one_hot(sample['label'], depth=10)

    if G.data_format == 'channels_first':
        image = tf.transpose(image, (2, 0, 1))

    return image, label


def extract_cifar10_raw_fn(tfrecord):
    features = {
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    }
    sample = tf.parse_single_example(tfrecord, features)
    image = tf.decode_raw(sample['image_raw'], tf.uint8)
    image = tf.reshape(image, [32, 32, 3])
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Normalize
    image = tf.subtract(image, _cifar_mean)
    image = tf.divide(image, _cifar_std)
    label = tf.one_hot(sample['label'], depth=10)

    if G.data_format == 'channels_first':
        image = tf.transpose(image, (2, 0, 1))

    return image, label

def total_num_records(tf_records_filenames):
    c = 0
    for fn in tf_records_filenames:
        for record in tf.python_io.tf_record_iterator(fn):
            c += 1
    return c
