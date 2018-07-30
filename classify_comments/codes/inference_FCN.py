# -*- coding: utf-8 -*-
# a fully connected network
import tensorflow as tf

# INPUT_NODE = 784
LAYER1_NODE = 2000
LAYER2_NODE = 2000
OUTPUT_NODE = 3


def __get_weights(shape):
    weights = tf.get_variable(
        "weight", 
        shape,
        tf.float32,
        tf.truncated_normal_initializer(
            stddev=0.1
        )
    )
    return weights


def __get_biases(shape):
    biases = tf.get_variable(
        "bias",
        shape,
        tf.float32,
        tf.constant_initializer(
            0.0
        )
    )
    return biases


# input_node here is the len of lex
def inference(input_tensor, input_node, regularizer):
    with tf.variable_scope("layer1"):
        weights1 = __get_weights([input_node, LAYER1_NODE])
        biases1 = __get_biases([LAYER1_NODE])
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
    with tf.variable_scope("layer2"):
        weights2 = __get_weights([LAYER1_NODE, LAYER2_NODE])
        biases2 = __get_biases([LAYER2_NODE])
        layer2 = tf.nn.relu(tf.matmul(layer1, weights2) + biases2)
    with tf.variable_scope("layer3"):
        weights3 = __get_weights([LAYER2_NODE, OUTPUT_NODE])
        biases3 = __get_biases([OUTPUT_NODE])
        layer3 =tf.matmul(layer2, weights3) + biases3
    return layer3
    




# # 28 (edge) * 28
# IMAGE_SIZE = 28
# # 黑白
# NUM_CHANNELS = 1
# NUM_LABELS = 10

# CONV1_DEEP = 32
# # 过滤器尺寸
# CONV1_SIZE = 5

# CONV2_DEEP = 64
# CONV2_SIZE = 5

# CONV3_DEEP = 64
# CONV3_SIZE = 5
# # num of Fully connected nodes
# FC_SIZE = 512


# def inference(input_tensor, train, regularizer):
#     # with tf.variable_scope('layer1-conv1'):
#     #     conv1_weights = tf.get_variable(  # 与 tf.Variable() 类似
#     #         "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],  # x, y, prev-depth, depth
#     #         initializer=tf.truncated_normal_initializer(stddev=0.1)
#     #     )
#     #     conv1_biases = tf.get_variable(
#     #         "bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0)
#     #     )

#     #     # 过滤器：边长5，深度32，移动步长1，填充全0 
#     #     conv1 = tf.nn.conv2d(
#     #         input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME'
#     #     )
#     #     relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

#     # # https://www.jianshu.com/p/cff8678de15a
#     # # 最大池化层：
#     # with tf.name_scope('layer2-pool1'):
#     #     # 过滤器：边长2，移动步长2，全0填充
#     #     pool1 = tf.nn.max_pool(
#     #         relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'
#     #     )

#     # with tf.variable_scope('layer3-conv2'):
#     #     conv2_weights = tf.get_variable(
#     #         "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
#     #         initializer=tf.truncated_normal_initializer(stddev=0.1)
#     #     )
#     #     conv2_biases = tf.get_variable(
#     #         "bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0)
#     #     )

#     #     conv2 = tf.nn.conv2d(
#     #         pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME'
#     #     )
#     #     relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

#     # with tf.name_scope('layer4-pool2'):
#     #     pool2 = tf.nn.max_pool(
#     #         relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'
#     #     )

#     # convolution + maxpool layer
#     num_filters = 128  # depth
#     filter_sizes = [3,4,5]
#     pooled_outputs = []
#     for i, filter_size in enumerate(filter_sizes):
#         with tf.name_scope("conv-maxpool-%s" % filter_size):
#             filter_shape = [filter_size, embedding_size, 1, num_filters]  # what is the num 1 for?
#             W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
#             b = tf.Variable(tf.constant(0.1, shape=[num_filters]))
#             conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID")
#             h = tf.nn.relu(tf.nn.bias_add(conv, b))
#             pooled = tf.nn.max_pool(h, ksize=[1, input_size - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
#             pooled_outputs.append(pooled)
 
#     num_filters_total = num_filters * len(filter_sizes)

#     # as_list 拉成向量
#     pool_shape = pool2.get_shape().as_list()

#     # pool_shape[0] 为一个batch中数据的个数; 7*7*64
#     nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

#     # reshaped = tf.reshape(pool2, [pool_shape[0], nodes])
#     reshaped = tf.reshape(pool2, [-1, nodes])

#     with tf.variable_scope('layer5-fc1'):
#         fc1_weights = tf.get_variable(
#             "weight", [nodes, FC_SIZE],
#             initializer=tf.truncated_normal_initializer(stddev=0.1)
#         )
#         if regularizer != None:
#             tf.add_to_collection('losses', regularizer(fc1_weights))
#         fc1_biases = tf.get_variable(
#             "bias", [FC_SIZE], 
#             initializer=tf.constant_initializer(0.1)
#         )

#         fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
#         if train: 
#             fc1 = tf.nn.dropout(fc1, 0.5)

#     with tf.variable_scope('layer6-fc2'):
#         fc2_weights = tf.get_variable(
#             "weight", [FC_SIZE, NUM_LABELS], 
#             initializer=tf.truncated_normal_initializer(stddev=0.1)
#         )
#         if regularizer != None:
#             tf.add_to_collection('losses', regularizer(fc2_weights))
#         fc2_biases = tf.get_variable(
#             "bias", [NUM_LABELS],
#             initializer=tf.constant_initializer(0.1)
#         )
#         logit = tf.matmul(fc1, fc2_weights) + fc2_biases

#     return logit

    # # embedding layer
    # with tf.device('/cpu:0'), tf.name_scope("embedding"):
    #     embedding_size = 128
    #     W = tf.Variable(tf.random_uniform([input_size, embedding_size], -1.0, 1.0))
    #     embedded_chars = tf.nn.embedding_lookup(W, X)
    #     embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
    # # convolution + maxpool layer
    # num_filters = 128
    # filter_sizes = [3,4,5]
    # pooled_outputs = []
    # # 3 couple of convolution-pooling layers
    # for i, filter_size in enumerate(filter_sizes):
    #     with tf.name_scope("conv-maxpool-%s" % filter_size):
    #         filter_shape = [filter_size, embedding_size, 1, num_filters]
    #         W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
    #         b = tf.Variable(tf.constant(0.1, shape=[num_filters]))
    #         conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID")
    #         h = tf.nn.relu(tf.nn.bias_add(conv, b))
    #         pooled = tf.nn.max_pool(h, ksize=[1, input_size - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
    #         # 这是啥玩意?
    #         pooled_outputs.append(pooled)
 
    # num_filters_total = num_filters * len(filter_sizes)
    # h_pool = tf.concat(3, pooled_outputs)
    # h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    # # dropout
    # with tf.name_scope("dropout"):
    #     h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
    # # output
    # with tf.name_scope("output"):
    #     W = tf.get_variable("W", shape=[num_filters_total, num_classes], initializer=tf.contrib.layers.xavier_initializer())
    #     b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
    #     output = tf.nn.xw_plus_b(h_drop, W, b)
        
    # return output
    


