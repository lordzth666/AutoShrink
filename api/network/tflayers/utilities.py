import tensorflow as tf
import numpy as np
from api.backend import G

def convert_to_legacy(data_format):
    if data_format == "channels_first":
        return "NCHW"
    else:
        return "NHWC"


def hard_swish(x):
    return x * tf.nn.relu6(x + 3.0) / 6.0

def quantize_linear(x):
    return tf.identity(x)

def apply_activation(tensor, activation='relu'):
    """
        Applying the activation function to the tensor
    :param tensor: Tensor of any shape.
    :param activation: activation type.
    :return: Tensor after applying the activation function.
    """
    assert G.BACKEND == "tensorflow"
    # Use ELU all the time
    if activation == 'relu':
        tensor = tf.nn.relu(tensor, name='activation')
    elif activation == 'relu6':
        tensor = tf.nn.relu6(tensor, name='activation')
    elif activation == 'elu':
        tensor = tf.nn.elu(tensor, name='activation')
    elif activation == 'leaky':
        tensor = tf.nn.leaky_relu(tensor, alpha=.2, name='activation')
    elif activation == 'swish':
        print("Using swish...")
        tensor = tf.nn.swish(tensor)
    elif activation == 'hswish':
        tensor = hard_swish(tensor)
    elif activation == 'softmax':
        tf.logging.info("softmax detected!!!!!!")
        tensor = tf.nn.softmax(tensor, name='activation')
    elif activation == 'sigmoid':
        tensor = tf.nn.sigmoid(tensor, name='activation')
    elif activation == 'linear' or activation is None:
        tensor = tensor
    else:
        raise NotImplementedError
    return tensor


def get_activation_fn(activation):
    return lambda tensor: apply_activation(tensor, activation)


def batch_normalization_v1(input,
                        activation='linear',
                        trainable=True,
                        is_training=False,
                        ):
    """
    Batch normalization layer supporting tflite. Fuse activation into batchnorm.
    :param inputs:
    :return:
    """
    activation_fn = get_activation_fn(activation)
    with tf.variable_scope("bn", tf.AUTO_REUSE) as scope:
        output = tf.contrib.layers.batch_norm(input,
                                              scale=True,
                                              data_format=convert_to_legacy(G.data_format),
                                              decay=G.BN_MOMENTUM,
                                              epsilon=G.BN_EPSILON,
                                              trainable=trainable,
                                              reuse=tf.AUTO_REUSE,
                                              scope=scope,
                                              activation_fn=activation_fn,
                                              is_training=is_training,
                                              fused=True)
    return output


def batch_normalization_v2(input,
                        activation='linear',
                        trainable=True,
                        is_training=False,
                        ):
    """
    Batch normalization layer supporting tflite. Fuse activation into batchnorm.
    :param inputs:
    :return:
    """
    activation_fn = get_activation_fn(activation)
    with tf.variable_scope("bn", tf.AUTO_REUSE) as scope:
        if G.data_format == 'channels_first':
            axis = 1
        else:
            axis = -1
        output = tf.layers.batch_normalization(input,
                                               scale=True,
                                               axis=axis,
                                               momentum=G.BN_MOMENTUM,
                                               epsilon=G.BN_EPSILON,
                                               trainable=trainable,
                                               reuse=tf.AUTO_REUSE,
                                               training=is_training,
                                               fused=True)
        output = activation_fn(output)
    return output

batch_normalization = batch_normalization_v1


def convolution2d(inputs,
                  kernel_size,
                  filters,
                  strides=1,
                  padding='VALID',
                  batchnorm=False,
                  activation='linear',
                  initializer=G.BACKEND_DEFAULT_CONV_INITIALIZER(),
                  bias_initializer=tf.constant_initializer(0.00),
                  regularizer=None,
                  trainable=True,
                  use_bias=True,
                  is_training=tf.convert_to_tensor(True),
                  mode=None):
    """
        Convolutional 2D Layer.
    :param inputs: A 4-D Tensor: [batch, height, width, ifm]
    :param kernel_size: an integer indicating the filter kernel size.
    :param filters: number of filters in the output.
    :param strides: an integer indicating the convolution strides.
    :param padding: Whether to use the 'SAME' padding or 'VALID' padding. case insensitive.
    :param batchnorm: Whether to use batchnorm.
    :param activation: activation function.
    :param initializer: Weight initializer. See backend for default settings.
    :param bias_initializer: Bias initializer.
    :param regularizer: Regularizer for weights. See backend for default settings.
    :param trainable: Whether this layer is trainable.
    :param use_bias: Use bias or not.
    :return: A 4-D tensor: [batch, height_, width_, ofm].
    """
    assert G.BACKEND == "tensorflow"
    if regularizer is None:
        regularizer = G.BACKEND_DEFAULT_REGULARIZER(G.DEFAULT_REG)
    else:
        regularizer = regularizer
    if filters == -1:
        filters = int(inputs.get_shape()[-1])
    activation_fn = get_activation_fn(activation)
    if mode == 'conv-bn-relu':
        conv = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            trainable=trainable,
            activation=None,
            kernel_initializer=initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer,
            data_format=G.data_format,
            name='conv'
        )

        if batchnorm:
            conv = batch_normalization(conv,
                                       activation=activation,
                                       trainable=trainable,
                                       is_training=is_training,
                                       )
        else:
            conv = activation_fn(conv)
    elif mode == "relu-conv-bn":
        raise DeprecationWarning("relu-conv-bn is deprecated. Please use the conv-bn-relu triplet.")
    else:
        raise NotImplementedError

    return conv


def concat(inputs,
           axis=-1,
           name='concat'):
    """
        Concat tensors of different shape.
    :param inputs: list of tensors to concatenate.
    :param axis: axis to concatenate the tensor.
    :param name: name of the concatenate operation.
    :return: List of the concatenation output.
    """
    if G.data_format == 'channels_first':
        axis = 1
    else:
        axis = -1
    outputs = tf.concat(inputs, axis=axis, name=name)
    return outputs


def convolution1d(inputs,
                  kernel_size,
                  filters,
                  strides=1,
                  padding='VALID',
                  batchnorm=False,
                  activation='linear',
                  initializer=G.BACKEND_DEFAULT_INITIALIZER(),
                  bias_initializer=tf.constant_initializer(0.00),
                  regularizer=G.BACKEND_DEFAULT_REGULARIZER(0.00),
                  trainable=True,
                  use_bias=True,
                  is_training=True
                  ):
    """
        Convolutional 1D Layer.
    :param inputs: A 3-D Tensor: [batch, feat_num, ifm]
    :param kernel_size: an integer indicating the filter kernel size.
    :param filters: number of filters in the output.
    :param strides: an integer indicating the convolution strides.
    :param padding: Whether to use the 'SAME' padding or 'VALID' padding. case insensitive.
    :param batchnorm: Whether to use batchnorm.
    :param activation: activation function.
    :param initializer: Weight initializer. See backend for default settings.
    :param bias_initializer: Bias initializer.
    :param regularizer: Regularizer for weights. See backend for default settings.
    :param trainable: Whether this layer is trainable.
    :param use_bias: Use bias or not.
    :return: A 3-D tensor: [batch, feat_num_, ofm].
    """
    assert G.BACKEND == "tensorflow"
    conv = tf.layers.conv1d(inputs=inputs,
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=padding,
                            kernel_initializer=initializer,
                            bias_initializer=bias_initializer,
                            kernel_regularizer=regularizer,
                            trainable=trainable,
                            name='conv',
                            use_bias=use_bias)
    if batchnorm:
        conv = tf.layers.batch_normalization(inputs=conv, epsilon=G.BN_EPSILON, momentum=G.BN_MOMENTUM,
                                             name='bn', trainable=trainable, fused=False,
                                             reuse=tf.AUTO_REUSE,
                                             training=is_training)
    # will apply activation after
    conv = apply_activation(conv, activation)
    return conv


def dense(inputs,
          units,
          activation='relu',
          batchnorm=False,
          initializer=G.BACKEND_DEFAULT_FC_INITIALIZER(),
          bias_initializer=tf.constant_initializer(0.00),
          regularizer=None,
          trainable=True,
          use_bias=True,
          is_training=tf.convert_to_tensor(True)):
    """
        Dense Connection Layer. (Fully connected Layer)
    :param inputs: A 2-D Tensor: [batch, ifm]
    :param units: output units number (ofm)
    :param activation: activation function to apply.
    :param batchnorm: Whether use batch normalization or not.
    :param initializer: Initialization for weight parameter. See backend for details.
    :param bias_initializer: Initialization for bias parameter. See backend for details.
    :param regularizer: regularization function.
    :param trainable: Whether this layer is trainable or not.
    :param use_bias: Whether to use bias or not.
    :param is_training: Whether is training.
    :return: A 2-D Tensor: [batch, ofm]
    """
    assert G.BACKEND == "tensorflow"
    if regularizer is None:
        regularizer = G.BACKEND_DEFAULT_REGULARIZER(G.DEFAULT_REG)
    else:
        regularizer = regularizer
    fc = tf.layers.dense(
        inputs=inputs,
        units=units,
        kernel_initializer=initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=regularizer,
        use_bias=use_bias,
        trainable=trainable,
    )


    if batchnorm:
        fc = batch_normalization(fc,
                                activation=activation,
                                trainable=trainable,
                                is_training=is_training,
                                )
    else:
        activation_fn = get_activation_fn(activation)
        fc = activation_fn(fc)
    return fc


def maxpool(inputs, strides=2, ksize=2, padding='SAME'):
    """
        MaxPooling Layer.
    :param inputs: A 4-D Tensor: [batch, height, width, ifm].
    :param strides: Strides for pooling.
    :param ksize: kernel size for pooling. (Pool size)
    :param padding: Use 'SAME' padding or 'VALID' padding for output.
    :return: A 4-D Tensor: [batch, height_, width_, ofm]
    """
    return tf.layers.max_pooling2d(pool_size=ksize,
                                   strides=strides, padding=padding, inputs=inputs,
                                   data_format=G.data_format)

def maxpool1D(inputs, strides=2, ksize=2, padding='SAME'):
    """
        MaxPooling 1D Layer.
    :param inputs: A 3-D Tensor: [batch, feat_num, ifm]
    :param strides: Strides for pooling.
    :param ksize: Kernel size for pooling.
    :param padding: Use 'SAME' padding or 'VALID' padding for output.
    :return: A 3-D Tensor: [batch, feat_num_, ofm]
    """
    return tf.layers.max_pooling1d(pool_size=ksize,
                          strides=strides, padding=padding, inputs=inputs)


def avgpool(inputs, strides=2, ksize=2, padding='SAME'):
    """
        Average Pooling layer
    :param inputs: A 4-D Tensor: [batch, height, width, ifm]
    :param strides: Strides for pooling.
    :param ksize: Kernel size for pooling.
    :param padding: Use 'SAME' padding or 'VALID' padding for output.
    :return: A 4-D Tensor: [batch, height, width, ifm]
    """
    return tf.layers.average_pooling2d(pool_size=ksize,
                          strides=strides, padding=padding, inputs=inputs,
                          data_format=G.data_format)


def avgpool1D(inputs, strides=2, ksize=2, padding='SAME'):
    """
        Average Pooling 1D layer
    :param inputs: A 4-D Tensor: [batch, height, width, ifm]
    :param strides: Strides for pooling.
    :param ksize: Kernel size for pooling.
    :param padding: Use 'SAME' padding or 'VALID' padding for output.
    :return: A 4-D Tensor: [batch, height_, width_, ifm]
    """
    return tf.layers.average_pooling1d(pool_size=ksize,
                                       strides=strides, padding=padding, inputs=inputs,
                                       data_format=G.data_format)


def globalAvgPool(inputs):
    """
        Global Average Pooling Layer.
    :param inputs: A 4-D tensor: [batch, height, width, ifm]
    :return: A 4-D Tensor: [batch, 1, 1, ifm]
    """
    if G.data_format == "channels_last":
        feat_size = int(inputs.get_shape()[1])
    else:
        feat_size = int(inputs.get_shape()[-1])
    return tf.layers.average_pooling2d(inputs=inputs, pool_size=feat_size,
                                       strides=feat_size,
                                       data_format=G.data_format)


def globalAvgPool1D(inputs):
    """
        Global Average Pooling Layer 1D.
    :param inputs: A 3-D Tensor: [batch, feat_num, ifm]
    :return: A 3-D Tensor: [batch, 1, ifm]
    """
    feat_size = int(inputs.get_shape()[1])
    return tf.layers.average_pooling1d(inputs=inputs, pool_size=feat_size,
                                       strides=feat_size)


def flatten(inputs):
    """
        Flatten layer
    :param inputs: A Tensor of any shape.
    :return: Flattened 1-D Tensor.
    """
    return tf.layers.flatten(inputs=inputs)


def yolo_loss(predicts,
              labels,
              objects_num,
              batch_size=32,
              cell_size=7,
              num_classes=20,
              image_size=224,
              object_scale=1.0,
              noobject_scale=0.5,
              class_scale=1.0,
              coord_scale=1.0,
              boxes_per_cell=2):
    """Add Loss to all the trainable variables
    Args:
      predicts: 4-D tensor [batch_size, cell_size, cell_size, 5 * boxes_per_cell]
      ===> (num_classes, boxes_per_cell, 4 * boxes_per_cell)
      labels  : 3-D tensor of [batch_size, max_objects, 5]
      objects_num: 1-D tensor [batch_size]
    """

    assert G.BACKEND == "tensorflow"

    def yolo_iou(boxes1, boxes2):
        """calculate ious
        Args:
          boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 1-D tensor [4] ===> (x_center, y_center, w, h)
        Return:
          iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        boxes1 = tf.stack([boxes1[:, :, :, 0] - boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] - boxes1[:, :, :, 3] / 2,
                          boxes1[:, :, :, 0] + boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] + boxes1[:, :, :, 3] / 2])
        boxes1 = tf.transpose(boxes1, [1, 2, 3, 0])
        boxes2 = tf.stack([boxes2[0] - boxes2[2] / 2, boxes2[1] - boxes2[3] / 2,
                          boxes2[0] + boxes2[2] / 2, boxes2[1] + boxes2[3] / 2])

        # calculate the left up point
        lu = tf.maximum(boxes1[:, :, :, 0:2], boxes2[0:2])
        rd = tf.minimum(boxes1[:, :, :, 2:], boxes2[2:])

        # intersection
        intersection = rd - lu
        inter_square = intersection[:, :, :, 0] * intersection[:, :, :, 1]

        mask = tf.cast(intersection[:, :, :, 0] > 0, tf.float32) * tf.cast(intersection[:, :, :, 1] > 0, tf.float32)

        inter_square = mask * inter_square

        # calculate the boxs1 square and boxs2 square
        square1 = (boxes1[:, :, :, 2] - boxes1[:, :, :, 0]) * (boxes1[:, :, :, 3] - boxes1[:, :, :, 1])
        square2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])

        return inter_square / (square1 + square2 - inter_square + 1e-6)

    def yolo_cond1(num,
                   object_num,
                   loss,
                   predict,
                   labels,
                   cls_acc):
        return num < object_num

    def yolo_body1(num,
                   object_num,
                   loss,
                   predict,
                   labels,
                   cls_acc):
        """
        calculate loss
        Args:
          predict: 3-D tensor [cell_size, cell_size, 5 * boxes_per_cell]
          labels : [max_objects, 5]  (x_center, y_center, w, h, class)
        """
        label = labels[num, :]
        label = tf.reshape(label, [5])
        # calculate objects  tensor [CELL_SIZE, CELL_SIZE]
        min_x = (label[0] - label[2] / 2) / (image_size / cell_size)
        max_x = (label[0] + label[2] / 2) / (image_size / cell_size)

        min_y = (label[1] - label[3] / 2) / (image_size / cell_size)
        max_y = (label[1] + label[3] / 2) / (image_size / cell_size)

        min_x = tf.floor(min_x)
        min_y = tf.floor(min_y)

        max_x = tf.ceil(max_x)
        max_y = tf.ceil(max_y)

        temp = tf.cast(tf.stack([max_y - min_y, max_x - min_x]), dtype=tf.int32)
        objects = tf.ones(temp, tf.float32)

        temp = tf.cast(tf.stack([min_y, cell_size - max_y, min_x, cell_size - max_x]), tf.int32)
        temp = tf.reshape(temp, (2, 2))
        objects = tf.pad(objects, temp, "CONSTANT")

        # calculate objects  tensor [CELL_SIZE, CELL_SIZE]
        # calculate responsible tensor [CELL_SIZE, CELL_SIZE]
        center_x = label[0] / (image_size / cell_size)
        center_x = tf.floor(center_x)

        center_y = label[1] / (image_size / cell_size)
        center_y = tf.floor(center_y)

        response = tf.ones([1, 1], tf.float32)

        temp = tf.cast(tf.stack([center_y, cell_size - center_y - 1, center_x, cell_size - center_x - 1]),
                       tf.int32)
        temp = tf.reshape(temp, (2, 2))
        response = tf.pad(response, temp, "CONSTANT")
        # objects = response

        # calculate iou_predict_truth [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        predict_boxes = predict[:, :, num_classes + boxes_per_cell:]

        predict_boxes = tf.reshape(predict_boxes, [cell_size, cell_size, boxes_per_cell, 4])

        predict_boxes = predict_boxes * [image_size / cell_size, image_size / cell_size,
                                         image_size, image_size]

        base_boxes = np.zeros([cell_size, cell_size, 4])

        for y in range(cell_size):
            for x in range(cell_size):
                base_boxes[y, x, :] = [image_size / cell_size * x, image_size / cell_size * y, 0, 0]
        base_boxes = np.tile(np.resize(base_boxes, [cell_size, cell_size, 1, 4]),
                             [1, 1, boxes_per_cell, 1])

        predict_boxes = base_boxes + predict_boxes

        iou_predict_truth = yolo_iou(predict_boxes, label[0:4])

        # calculate C [cell_size, cell_size, boxes_per_cell]
        C = iou_predict_truth * tf.reshape(response, [cell_size, cell_size, 1])

        # calculate I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        I = iou_predict_truth * tf.reshape(response, (cell_size, cell_size, 1))

        max_I = tf.reduce_max(I, 2, keepdims=True)

        I = tf.cast((I >= max_I), tf.float32) * tf.reshape(response, (cell_size, cell_size, 1))

        # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        no_I = tf.ones_like(I, dtype=tf.float32) - I

        p_C = predict[:, :, num_classes:num_classes + boxes_per_cell]

        # calculate truth x,y,sqrt_w,sqrt_h 0-D
        x = label[0]
        y = label[1]

        sqrt_w = tf.sqrt(tf.abs(label[2]))
        sqrt_h = tf.sqrt(tf.abs(label[3]))

        # calculate predict p_x, p_y, p_sqrt_w, p_sqrt_h 3-D [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        p_x = predict_boxes[:, :, :, 0]
        p_y = predict_boxes[:, :, :, 1]

        p_sqrt_w = tf.sqrt(tf.minimum(image_size * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 2])))
        p_sqrt_h = tf.sqrt(tf.minimum(image_size * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 3])))
        # calculate truth p 1-D tensor [NUM_CLASSES]
        P = tf.one_hot(tf.cast(label[4], tf.int32), num_classes, dtype=tf.float32)

        # calculate predict p_P 3-D tensor [CELL_SIZE, CELL_SIZE, NUM_CLASSES]
        p_P = predict[:, :, 0:num_classes]

        # calculate the classification precision

        # calculate the ground-truth class
        gt_cls = tf.argmax(P, -1)
        pred_cls = tf.argmax(p_P, -1)
        # calculate cls accuracy
        acc = tf.reduce_mean(tf.cast(tf.equal(gt_cls, pred_cls), tf.float32))
        cls_acc += acc / tf.cast(object_num, tf.float32)

        # class loss
        class_loss = tf.nn.l2_loss(
            tf.reshape(objects, (cell_size, cell_size, 1)) * (p_P - P)) * class_scale


        # object_loss
        object_loss = tf.nn.l2_loss(I * (p_C - C)) * object_scale
        # object_loss = tf.nn.l2_loss(I * (p_C - (C + 1.0)/2.0)) * object_scale

        # noobject_loss
        noobject_loss = tf.nn.l2_loss(no_I * (p_C)) * noobject_scale

        # coord_loss
        coord_loss = (tf.nn.l2_loss(I * (p_x - x) / (image_size / cell_size)) +
                      tf.nn.l2_loss(I * (p_y - y) / (image_size / cell_size)) +
                      tf.nn.l2_loss(I * (p_sqrt_w - sqrt_w)) / image_size +
                      tf.nn.l2_loss(I * (p_sqrt_h - sqrt_h)) / image_size) * coord_scale

        return num + 1, object_num, [loss[0] + class_loss, loss[1] + object_loss, loss[2] + noobject_loss,
                                     loss[3] + coord_loss], predict, labels, cls_acc

    class_loss = tf.constant(0, tf.float32)
    object_loss = tf.constant(0, tf.float32)
    noobject_loss = tf.constant(0, tf.float32)
    coord_loss = tf.constant(0, tf.float32)

    loss = tf.zeros(shape=[4], dtype=tf.float32)
    avg_cls_acc = tf.constant(0, tf.float32)

    for i in range(batch_size):
        predict = predicts[i, :, :, :]
        label = labels[i, :, :]
        object_num = objects_num[i]
        tuple_results = tf.while_loop(yolo_cond1, yolo_body1,
                                       [tf.constant(0, dtype=tf.int32),
                                        object_num,
                                        [class_loss, object_loss, noobject_loss, coord_loss],
                                        predict,
                                        label,
                                        tf.constant(0, tf.float32)])

        loss = loss + tuple_results[2]
        avg_cls_acc += tuple_results[5] / batch_size



    with tf.device("/cpu:0"):
        tf.summary.scalar('yolo_loss/total_loss', (loss[0] + loss[1] + loss[2] + loss[3]) / batch_size)
        tf.summary.scalar('yolo_loss/class_loss', loss[0] / batch_size)
        tf.summary.scalar('yolo_loss/object_loss', loss[1] / batch_size)
        tf.summary.scalar('yolo_loss/noobject_loss', loss[2] / batch_size)
        tf.summary.scalar('yolo_loss/coord_loss', loss[3] / batch_size)

    total_loss = (loss[0] + loss[1] + loss[2] + loss[3]) / batch_size

    return total_loss, avg_cls_acc

def depthwise_conv2d(inputs,
                     kernel_size,
                     strides=1,
                     padding='SAME',
                     batchnorm=False,
                     depth_multiplier=1,
                     activation='linear',
                     initializer=G.BACKEND_DEFAULT_CONV_INITIALIZER(),
                     bias_initializer=tf.constant_initializer(0.00),
                     regularizer=G.BACKEND_DEFAULT_REGULARIZER(0.00),
                     trainable=True,
                     use_bias=True,
                     is_training=tf.convert_to_tensor(True),
                     mode=G.EXEC_CONV_MODE
                     ):
    """
        Depthwise Convolution
    :param inputs: A 4-D Tensor: [batch, height, width, ifm]
    :param kernel_size: Kernel size for depthwise Convolution.
    :param strides: Strides for depthwise convolution.
    :param padding: Padding for depthwise convolution.
    :param batchnorm: Whether use batchnorm or not.
    :param depth_multiplier: Multiplier for depthwise convolution.
    :param activation: Activation function for depthwise Convolution/
    :param initializer: Intializer for depthwise kernel.
    :param bias_initializer: Initializer for bias kernel.
    :param regularizer: Regularization for depthwise kernel.
    :param trainable: Whether this layer is trainable or not.
    :param use_bias: Whether to use bias or not.
    :return:
    """
    assert G.BACKEND == "tensorflow"
    if G.data_format == "channels_last":
        input_dim = inputs.get_shape()[-1]
        strides_ = (1, strides, strides, 1)
    else:
        input_dim = inputs.get_shape()[1]
        strides_ = (1, 1, strides, strides)


    output_dim = input_dim * depth_multiplier
    if G.data_format == 'channels_first':
        weights = tf.get_variable(shape=[kernel_size, kernel_size, input_dim, depth_multiplier], name="depthwise_kernel",
                                  initializer=initializer, regularizer=regularizer, trainable=trainable)
    else:
        weights = tf.get_variable(shape=[kernel_size, kernel_size, input_dim, depth_multiplier], name="depthwise_kernel",
                                  initializer=initializer, regularizer=regularizer, trainable=trainable)
    activation_fn = get_activation_fn(activation)

    if mode == 'conv-bn-relu':
        if use_bias:
            bias = tf.get_variable(shape=[output_dim], initializer=bias_initializer, name='bias', trainable=trainable)
            conv_depthwise = tf.nn.bias_add(tf.nn.depthwise_conv2d(input=inputs,
                                                                   filter=weights,
                                                                   strides=strides_,
                                                                   padding=padding,
                                                                   data_format=convert_to_legacy(G.data_format)), bias, name="depthwise_conv")
        else:
            conv_depthwise = tf.nn.depthwise_conv2d(input=inputs,
                                                    filter=weights,
                                                    strides=strides_,
                                                    padding=padding,
                                                    name="depthwise_conv",
                                                    data_format=convert_to_legacy(G.data_format))
        if batchnorm:
            with tf.variable_scope("depthwise_bn", tf.AUTO_REUSE):
                conv_depthwise = batch_normalization(conv_depthwise,
                                           activation=activation,
                                           trainable=trainable,
                                           is_training=is_training,
                                           )
        else:
            conv_depthwise = activation_fn(conv_depthwise)
    elif mode == 'relu-conv-bn':
        raise DeprecationWarning("relu-conv-bn is deprecated. Please use the conv-bn-relu triplet.")
    else:
        raise NotImplementedError
    return conv_depthwise

def pointwise_conv2d(inputs,
                     filters,
                     strides=1,
                     padding='SAME',
                     batchnorm=False,
                     activation='linear',
                     initializer=G.BACKEND_DEFAULT_CONV_INITIALIZER(),
                     bias_initializer=tf.constant_initializer(0.00),
                     regularizer=G.BACKEND_DEFAULT_REGULARIZER(0.00),
                     trainable=True,
                     use_bias=True,
                     is_training=tf.convert_to_tensor(True),
                     mode=G.EXEC_CONV_MODE
                     ):

    assert G.BACKEND == "tensorflow"
    input_dim = inputs.get_shape()[-1]
    output_dim = filters
    weights = tf.get_variable(shape=[1, 1, input_dim, output_dim], name="pointwise_kernel",
                              initializer=initializer, regularizer=regularizer, trainable=trainable)
    activation_fn = get_activation_fn(activation)

    if G.data_format == "channels_last":
        strides_ = (1, strides, strides, 1)
    else:
        strides_ = (1, 1, strides, strides)

    if mode == 'conv-bn-relu':
        if use_bias:
            bias = tf.get_variable(shape=[output_dim], initializer=bias_initializer, name='bias', trainable=trainable)
            conv_pointwise = tf.nn.bias_add(tf.nn.conv2d(input=inputs,
                                                         filter=weights,
                                                         strides=strides_,
                                                         padding=padding), bias, name="pointwise_conv")
        else:
            conv_pointwise = tf.nn.conv2d(input=inputs,
                                          filter=weights,
                                          strides=strides_,
                                          padding=padding,
                                          name='pointwise_conv')
        if batchnorm:
            with tf.variable_scope("pointwise_bn", tf.AUTO_REUSE):
                conv_pointwise = batch_normalization(conv_pointwise,
                                           activation=activation,
                                           trainable=trainable,
                                           is_training=is_training,
                                           )
        else:
            conv_pointwise = activation_fn(conv_pointwise)
    elif mode == 'relu-conv-bn':
        raise DeprecationWarning("relu-conv-bn is deprecated. Please use the conv-bn-relu triplet.")
    else:
        raise NotImplementedError
    return conv_pointwise


def separable_conv2d_v1(inputs,
                     kernel_size,
                     filters,
                     strides=1,
                     padding='SAME',
                     batchnorm=False,
                     activation='linear',
                     initializer=G.BACKEND_DEFAULT_CONV_INITIALIZER(),
                     bias_initializer=tf.constant_initializer(0.00),
                     regularizer=G.BACKEND_DEFAULT_REGULARIZER(0.00),
                     trainable=True,
                     regularize_depthwise=False,
                     use_bias=True,
                     is_training=tf.convert_to_tensor(True),
                     mode=G.EXEC_CONV_MODE
                     ):
    """
        Depthwise Separable Convolution Layer.
    :param inputs: A 4-D Tensor: [batch, height, width, ifm]
    :param kernel_size: an integer indicating the filter kernel size.
    :param filters: number of filters in the output.
    :param strides: an integer indicating the convolution strides.
    :param padding: Whether to use the 'SAME' padding or 'VALID' padding. case insensitive.
    :param batchnorm: Whether to use batchnorm.
    :param depthwise_activation: activation function for depthwise convolution.
    :param pointwise_activation: activation function for pointwise convolution.
    :param initializer: Weight initializer. See backend for default settings.
    :param bias_initializer: Bias initializer.
    :param regularizer: Regularizer for weights. See backend for default settings.
    :param trainable: Whether this layer is trainable.
    :param use_bias: Use bias or not.
    :return: A 4-D tensor: [batch, height_, width_, ofm].
    """
    if filters == -1:
        filters = int(inputs.get_shape()[-1])
    assert G.BACKEND == "tensorflow"

    if activation == "linear":
        depthwise_activation = G.SEPCONV_V1_DEFAULT_ACTIVATION
    else:
        depthwise_activation = activation

    if regularize_depthwise:
        depthwise_regularizer = regularizer
    else:
        depthwise_regularizer = None
    with tf.variable_scope("depthwise", tf.AUTO_REUSE) as scope:
        # Consistent with separable_conv_v2:
        with tf.variable_scope("separable_conv2d", tf.AUTO_REUSE) as scope:
            conv_depthwise = depthwise_conv2d(inputs=inputs,
                                              kernel_size=kernel_size,
                                              strides=strides,
                                              padding=padding,
                                              activation=depthwise_activation,
                                              depth_multiplier=1,
                                              initializer=initializer,
                                              bias_initializer=bias_initializer,
                                              regularizer=depthwise_regularizer,
                                              batchnorm=batchnorm,
                                              trainable=trainable,
                                              use_bias=use_bias,
                                              is_training=is_training,
                                              mode=mode)

            conv_pointwise = pointwise_conv2d(conv_depthwise,
                                              filters=filters,
                                              strides=1,
                                              activation=activation,
                                              initializer=initializer,
                                              bias_initializer=bias_initializer,
                                              regularizer=regularizer,
                                              padding='SAME',
                                              trainable=trainable,
                                              use_bias=use_bias,
                                              is_training=is_training,
                                              batchnorm=batchnorm,
                                              mode=mode)

    return conv_pointwise


def separable_conv2d_v2(inputs,
                     kernel_size,
                     filters,
                     strides=1,
                     padding='SAME',
                     batchnorm=False,
                     activation='linear',
                     initializer=G.BACKEND_DEFAULT_CONV_INITIALIZER(),
                     bias_initializer=tf.constant_initializer(0.00),
                     regularizer=G.BACKEND_DEFAULT_REGULARIZER(0.00),
                     trainable=True,
                     regularize_depthwise=False,
                     use_bias=True,
                     is_training=tf.convert_to_tensor(True),
                     mode=G.EXEC_CONV_MODE
                     ):
    """
        Depthwise Separable Convolution Layer.
    :param inputs: A 4-D Tensor: [batch, height, width, ifm]
    :param kernel_size: an integer indicating the filter kernel size.
    :param filters: number of filters in the output.
    :param strides: an integer indicating the convolution strides.
    :param padding: Whether to use the 'SAME' padding or 'VALID' padding. case insensitive.
    :param batchnorm: Whether to use batchnorm.
    :param depthwise_activation: activation function for depthwise convolution.
    :param pointwise_activation: activation function for pointwise convolution.
    :param initializer: Weight initializer. See backend for default settings.
    :param bias_initializer: Bias initializer.
    :param regularizer: Regularizer for weights. See backend for default settings.
    :param trainable: Whether this layer is trainable.
    :param use_bias: Use bias or not.
    :return: A 4-D tensor: [batch, height_, width_, ofm].
    """
    if filters == -1:
        if G.data_format == 'channels_last':
            filters = int(inputs.get_shape()[-1])
        else:
            filters = int(inputs.get_shape()[1])
    assert G.BACKEND == "tensorflow"
    if regularize_depthwise:
        depthwise_regularizer = regularizer
    else:
        depthwise_regularizer = None
    if mode == "conv-bn-relu":
        with tf.variable_scope("depthwise") as scope:
            conv = tf.layers.separable_conv2d(
                inputs=inputs,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                activation=None,
                depthwise_regularizer=depthwise_regularizer,
                pointwise_regularizer=regularizer,
                depthwise_initializer=initializer,
                pointwise_initializer=initializer,
                use_bias=use_bias,
                data_format=G.data_format
            )

        if batchnorm:
            conv = batch_normalization(conv,
                                       trainable=trainable,
                                       is_training=is_training,
                                       activation=activation
                                       )
    elif mode == 'relu-conv-bn':
        raise DeprecationWarning("relu-conv-bn is deprecated. Please use the conv-bn-relu triplet.")
    else:
        raise NotImplementedError


    return conv

separable_conv2d = separable_conv2d_v1
# You can use 'separable_conv2d_v2' for faster searching/evaluation, however,
# the architecture derived from 'separable_conv2d_v1' and 'separable_conv2d_v2' can be completely different,
# thus you may need to rerun the search.

def label_smoothing(inputs, epsilon=0.1):
    """
    Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
    :param inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
    :param epsilon: Smoothing rate.
    :return:
    """
    V = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / V)

