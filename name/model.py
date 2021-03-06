import tensorflow as tf
import numpy as np
import process, const


def model(input_tensor, input_size, vocabulary_size, dropout_keep_prob):



    # x = tf.placeholder(tf.int32, [None, input_size])
    # y_ = tf.placeholder(tf.float32, [None, num_classes])

    # dropout_keep_prob = tf.placeholder(tf.float32)
    return neural_network(input_tensor, input_size, vocabulary_size, dropout_keep_prob)



# input_tensor is a batch of id_lists (ex: [[1, 3, 2, 0, 0, 0, 0, 0], ...]), while input_size is its length (here is 8). Vocabulary_size is length of the vocab_list
def neural_network(input_tensor, input_size, vocabulary_size, dropout_keep_prob, embedding_size=128, num_filters=128):
	# embedding layer
	with tf.device('/cpu:0'), tf.name_scope("embedding"):
		W = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))  # trainable and randomly initiated
		embedded_chars = tf.nn.embedding_lookup(W, input_tensor)  # shape: [batches, input_size, embedding_size]
		embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)  # shape: [batches, input_size, embedding_size, 1]
	# convolution + maxpool layer
	filter_sizes = [3,4,5]
	pooled_outputs = []
	for i, filter_size in enumerate(filter_sizes):  # 生成并列的 3 * 128 个 m = 3, 4, 5不同的, n = 128 (embedding_size) 相同的 filters (m * n); 但是示例上提到的是生成 6 个
		with tf.name_scope("conv-maxpool-%s" % filter_size):
			filter_shape = [filter_size, embedding_size, 1, num_filters]  # conv layer 的 深度 是 128
			W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
			b = tf.Variable(tf.constant(0.1, shape=[num_filters]))
			# what is valid-padding? https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
			conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID")
			h = tf.nn.relu(tf.nn.bias_add(conv, b))
			pooled = tf.nn.max_pool(h, ksize=[1, input_size - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')  # 把向量转化为 1 * 1
			pooled_outputs.append(pooled)

	num_filters_total = num_filters * len(filter_sizes)  # num_filters here is the depth of the last pooling layer
	h_pool = tf.concat(pooled_outputs, 3)  # num 3 here references the num of filter
	h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
	# dropout
	with tf.name_scope("dropout"):
		h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
	# output
	with tf.name_scope("output"):
		W = tf.get_variable("W", shape=[num_filters_total, const.NUM_CLASSES], initializer=tf.contrib.layers.xavier_initializer())
		b = tf.Variable(tf.constant(0.1, shape=[const.NUM_CLASSES]))
		output = tf.nn.xw_plus_b(h_drop, W, b)

	return output
