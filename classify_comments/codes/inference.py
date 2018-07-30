import tensorflow as tf 
import const, process

# input_size = len(lex)
# input_size = process.get_lex_len()
OUTPUT_NODE = num_classes = 3

# X = tf.placeholder(tf.int32, [None, input_size])
# Y = tf.placeholder(tf.float32, [None, num_classes])
 
# dropout_keep_prob = tf.placeholder(tf.float32)
 
# batch_size = 90
 
def inference(input_tensor, input_size, is_train):
    if is_train is not None:
        dropout_keep_prob = 0.5
        logger = process.get_logger()
        logger.debug("dropout: %g" % dropout_keep_prob)
    else:
        dropout_keep_prob = 1.0
    # embedding layer
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
        embedding_size = 128
        W = tf.Variable(tf.random_uniform([input_size, embedding_size], minval=-1.0, maxval=1.0), name="embedding")
        embedded_chars = tf.nn.embedding_lookup(W, input_tensor)
        # embedded_chars = tf.nn.dropout(embedded_chars, dropout_keep_prob)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
    # convolution + maxpool layer
    num_filters = 128
    filter_sizes = [3,4,5]
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]))
            conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID")
            h = tf.nn.relu(tf.nn.bias_add(conv, b))
            pooled = tf.nn.max_pool(h, ksize=[1, input_size - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
            pooled_outputs.append(pooled)
 
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    # dropout
    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
    # output
    with tf.name_scope("output"):
        W = tf.get_variable("W", shape=[num_filters_total, num_classes], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
        output = tf.nn.xw_plus_b(h_drop, W, b)
        
    return output