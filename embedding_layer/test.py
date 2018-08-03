import numpy as np
import tensorflow as tf


def test():
    a = np.array(
        [[1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]]
    )
    # weights = tf.placeholder(tf.float32, [None, None])
    input_ids = [0, 2, 0,]

    # embedding_layer = tf.nn.embedding_lookup(weights, input_ids)
    embedding_layer1 = tf.nn.embedding_lookup(a, input_ids)

    with tf.Session() as sess:
        print(sess.run(embedding_layer1))


test()

# >>>
# [[ 1  2  3  4]
#  [ 9 10 11 12]
#  [ 1  2  3  4]]
