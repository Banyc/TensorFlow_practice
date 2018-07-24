# A customized model

import tensorflow as tf 
import const

# IMAGE_SIZE = 28
NUM_CHANNELS = 1

CONV1_SIZE = 3
CONV1_DEEP = 32

CONV2_SIZE = 3
CONV2_DEEP = 64

CONV3_SIZE = 3
CONV3_DEEP = 64

FC_SIZE = 1024

NUM_LABELS = const.MAX_CAPTCHA * const.CHAR_SET_LEN


def inference(input_tensor, Is_train, regularizer):
    with tf.variable_scope("layer1-conv1"):
        conv1_weights = tf.get_variable(
            "weight", 
            [CONV1_SIZE,
            CONV1_SIZE,
            NUM_CHANNELS,
            CONV1_DEEP],
            tf.float32,
            tf.truncated_normal_initializer(
                stddev=0.1
            )
        )
        conv1_biases = tf.get_variable(
            "bias",
            [CONV1_DEEP],
            tf.float32,
            tf.constant_initializer(
                0.0
            )
        )
        conv1 = tf.nn.conv2d(
            input_tensor, conv1_weights,
            [1, 1, 1, 1], "SAME"
        )
        relu1 = tf.nn.relu(
            tf.nn.bias_add(conv1, conv1_biases)
        )

    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(
            relu1, 
            ksize=[1, 2, 2, 1], 
            strides=[1, 2, 2, 1],
            padding="SAME"
        )

    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable(
            "weight", 
            [CONV2_SIZE,
            CONV2_SIZE,
            CONV1_DEEP,
            CONV2_DEEP],
            tf.float32,
            tf.truncated_normal_initializer(
                stddev=0.1
            )
        )
        conv2_biases = tf.get_variable(
            "bias", 
            [CONV2_DEEP],
            tf.float32,
            tf.constant_initializer(
                0.0
            )
        )
        conv2 = tf.nn.conv2d(
            pool1, conv2_weights,
            [1, 1, 1, 1], "SAME"
        )
        relu2 = tf.nn.relu(
            tf.nn.bias_add(conv2, conv2_biases)
        )

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(
            relu2, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], 
            padding="SAME"
        )

    with tf.variable_scope("layer5-conv3"):
        conv3_weights = tf.get_variable(
            "weight", 
            [CONV3_SIZE,
            CONV3_SIZE,
            CONV2_DEEP,
            CONV3_DEEP],
            tf.float32,
            tf.truncated_normal_initializer(
                stddev=0.1
            )
        )
        conv3_biases = tf.get_variable(
            "bias", 
            [CONV3_DEEP],
            tf.float32,
            tf.constant_initializer(
                0.0
            )
        )
        conv3 = tf.nn.conv2d(
            pool1, conv2_weights,
            [1, 1, 1, 1], "SAME"
        )
        relu3 = tf.nn.relu(
            tf.nn.bias_add(conv3, conv3_biases)
        )

    with tf.name_scope("layer6-pool3"):
        pool3 = tf.nn.max_pool(
            relu3, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], 
            padding="SAME"
        )

    pool_shape = pool3.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(
        pool3, 
        [-1,
        nodes]
    )
    with tf.variable_scope("layer7-fc1"):
        fc1_weights = tf.get_variable(
            "weight", 
            [nodes, FC_SIZE],
            tf.float32,
            tf.truncated_normal_initializer(
                stddev=0.1
            )
        )
        if regularizer != None:
            regularization = regularizer(fc1_weights)
            tf.add_to_collection("losses", regularization)
        fc1_biases = tf.get_variable(
            "bias",
            [FC_SIZE],
            tf.float32,
            tf.constant_initializer(
                0.1
            )
        )
        fc1 = tf.nn.relu(
            tf.matmul(reshaped, fc1_weights) + fc1_biases
        )
        if Is_train:
            fc1 = tf.nn.dropout(fc1, 0.75)

    with tf.variable_scope("layer8-fc2"):
        fc2_weights = tf.get_variable(
            "weight",
            [FC_SIZE, NUM_LABELS],
            tf.float32,
            tf.truncated_normal_initializer(
                stddev=0.1
            )
        )
        if regularizer != None:
            regularization = regularizer(fc2_weights)
            tf.add_to_collection("losses", regularization)
        fc2_biases = tf.get_variable(
            "bias",
            [NUM_LABELS],
            tf.float32,
            tf.constant_initializer(
                0.1
            )
        )
        logits = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logits