# -*- coding: utf-8 -*-
# converted from fully connected network version
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np

import mnist_inference

BATCH_SIZE = 100
# LEARNING_RATE_BASE = 0.8
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
# MODEL_SAVE_PATH = "model"
MODEL_SAVE_PATH = "./LeNet-5/model"
MODEL_NAME = "model.ckpt"


def train(mnist):
    x = tf.placeholder(
        tf.float32, 
        [BATCH_SIZE,
        mnist_inference.IMAGE_SIZE,
        mnist_inference.IMAGE_SIZE,
        mnist_inference.NUM_CHANNELS],
        name="x-input")
    y_ = tf.placeholder(
        tf.float32, [None, mnist_inference.NUM_LABELS], name="y_-input")

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, True, regularizer)
    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(
        tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)
    # learning_rate = tf.train.exponential_decay(
    #     LEARNING_RATE_BASE,
    #     global_step,
    #     mnist.train.num_examples / BATCH_SIZE,
    #     LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
                    .minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            # New
            reshaped_xs = np.reshape(xs, 
                (BATCH_SIZE, 
                mnist_inference.IMAGE_SIZE, 
                mnist_inference.IMAGE_SIZE,
                mnist_inference.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: reshaped_xs, y_: ys})
            # if i % 1000 == 0:
            if i % 50 == 0:
                print("After %d training step(s), loss on training "
                      "batch is %g." % (step, loss_value))
                saver.save(
                    sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                    global_step=global_step)


def main(argv=None):
    # mnist = input_data.read_data_sets("mnist_data", one_hot=True)
    mnist = input_data.read_data_sets("mnist", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()

# OUTPUT:
# After 1 training step(s), loss on training batch is 5.53604.
# After 1001 training step(s), loss on training batch is 3.07106.
# After 2001 training step(s), loss on training batch is 2.97472.
# After 3001 training step(s), loss on training batch is 2.85732.
# After 4001 training step(s), loss on training batch is 2.79748.
# After 5001 training step(s), loss on training batch is 2.69474.
# After 6001 training step(s), loss on training batch is 2.67005.
# After 7001 training step(s), loss on training batch is 2.61361.
# After 8001 training step(s), loss on training batch is 2.57829.
# After 9001 training step(s), loss on training batch is 2.54082.
# After 10001 training step(s), loss on training batch is 2.5121.
# After 11001 training step(s), loss on training batch is 2.47847.
# After 12001 training step(s), loss on training batch is 2.46153.
# After 13001 training step(s), loss on training batch is 2.44125.
# After 14001 training step(s), loss on training batch is 2.42775.
# After 15001 training step(s), loss on training batch is 2.40342.
# After 16001 training step(s), loss on training batch is 2.39694.
# After 17001 training step(s), loss on training batch is 2.39279.
# After 18001 training step(s), loss on training batch is 2.38261.
# After 19001 training step(s), loss on training batch is 2.37112.
# After 20001 training step(s), loss on training batch is 2.35535.
# After 21001 training step(s), loss on training batch is 2.34713.
# After 22001 training step(s), loss on training batch is 2.37502.
# After 23001 training step(s), loss on training batch is 2.33871.
# After 24001 training step(s), loss on training batch is 2.34273.
# After 25001 training step(s), loss on training batch is 2.3321.
# After 26001 training step(s), loss on training batch is 2.33264.
# After 27001 training step(s), loss on training batch is 2.33516.
# After 28001 training step(s), loss on training batch is 2.32879.
# After 29001 training step(s), loss on training batch is 2.33042.

# train: True->False
# After 1 training step(s), loss on training batch is 5.52516.
# After 1001 training step(s), loss on training batch is 9.66582e+12.
# After 2001 training step(s), loss on training batch is 8.2723e+12.
