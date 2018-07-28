# -*- coding: utf-8 -*-

import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np

import logging

import inference, const, process, evaluate

# BATCH_SIZE = 100
# # LEARNING_RATE_BASE = 0.8
# LEARNING_RATE_BASE = 0.01
# LEARNING_RATE_DECAY = 0.99
# REGULARIZATION_RATE = 0.0001
# TRAINING_STEPS = 30000
# MOVING_AVERAGE_DECAY = 0.99



def train():
    logger = process.get_logger()
    x = tf.placeholder(
        tf.int32, 
        [None,
        None],
        name="x-input")
    y_ = tf.placeholder(
        tf.float32, [None, inference.OUTPUT_NODE], name="y_-input")

    # the last accuracy, for evaluate
    pre_acc = tf.Variable(0.0, False, name="pre_accuracy")

    # regularizer = tf.contrib.layers.l2_regularizer(const.REGULARIZATION_RATE)
    y = inference.inference(x, process.get_lex_len(), None)
    global_step = tf.Variable(0, trainable=False, name="global_step")

    # variable_averages = tf.train.ExponentialMovingAverage(
    #     MOVING_AVERAGE_DECAY, global_step)
    # variable_averages_op = variable_averages.apply(
    #     tf.trainable_variables())
    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     logits=y, labels=tf.argmax(y_, 1))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=y, labels=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean #+ tf.add_n(tf.get_collection("losses"))
    # learning_rate = tf.train.exponential_decay(
    #     LEARNING_RATE_BASE,
    #     global_step,
    #     mnist.train.num_examples / BATCH_SIZE,
    #     LEARNING_RATE_DECAY)
    learning_rate = const.LEARNING_RATE_BASE
    # train_step = tf.train.GradientDescentOptimizer(learning_rate)\
    train_step = tf.train.AdamOptimizer(learning_rate)\
                    .minimize(loss, global_step=global_step)
    # with tf.control_dependencies([train_step, variable_averages_op]):
    #     train_op = tf.no_op(name="train")

    # for evaluate
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        ckpt = tf.train.get_checkpoint_state(
            const.MODEL_DIR
        )
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        # writer = tf.summary.FileWriter(const.TENSORBOARD_LOG_DIR, tf.get_default_graph())

        # for evaluate
        test_xs, test_ys = process.get_test_dataset()

        is_first_round = True
        # for i in range(const.TRAINING_STEPS):
        while True:
            xs, ys = process.get_next_batch(40)
            _, loss_value, step = sess.run([train_step, loss, global_step],
                                           feed_dict={x: xs, y_: ys})

            if is_first_round:
                cur_acc = sess.run(accuracy, {x: test_xs[0:50], y_: test_ys[0:50]})  # 过大会内存爆炸
                logger.info("After the first train, %s training step(s), validation "
                    "accuracy = %g" % (step, cur_acc))
                logger.info("While pre_accuracy is: %g" % pre_acc.eval())
                is_first_round = False

            # if i % 1000 == 0:
            # if i % 50 == 0:
            # if step % 50 == 0:
            if step % 2 == 0:
                print("After %d training step(s), loss on training "
                      "batch is %g." % (step, loss_value))
                logger.info("After %d training step(s), loss on training "
                    "batch is %g." % (step, loss_value))
                
                # for evaluate
                # current_acc
                cur_acc, test_loss = sess.run((accuracy, loss), {x: test_xs[0:50], y_: test_ys[0:50]})
                logger.info("After %s training step(s), validation "
                    "accuracy = %g, loss = %g" % (step, cur_acc, test_loss))
                logger.info("While pre_accuracy is: %g" % pre_acc.eval())
                if cur_acc >= pre_acc.eval():
                    saver.save(
                    sess, os.path.join(const.MODEL_DIR, const.MODEL_NAME),
                    global_step=global_step)
                    sess.run(tf.assign(pre_acc, cur_acc))
                if cur_acc > 0.8:
                    break
                

def main(argv=None):
    logger = process.get_logger()
    logger.info("Training (CNN) started")
    # logger.info("Training step: %g", const.TRAINING_STEPS)
    logger.info("Learning rate: %g", const.LEARNING_RATE_BASE)
    
    train()
    logger.info("Training (CNN) ended")

    return 0


if __name__ == '__main__':
    tf.app.run()
