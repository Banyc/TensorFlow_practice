# -*- coding: utf-8 -*-
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np

import const, process, inference

EVAL_INTERVAL_SECS = 10


def evaluate():
    logger = process.get_logger()
    
    with tf.Graph().as_default() as g:

        xs, ys = process.get_test_dataset()


        x = tf.placeholder(
            tf.int32,
            [None,
            None],
            name="x-input"
        )
        y_ = tf.placeholder(
            tf.float32, [None, inference.OUTPUT_NODE], name="y_-input")

        validate_feed = {x: xs,
                         y_:ys}

        y = inference.inference(x, process.get_lex_len(), None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # variable_averages = tf.train.ExponentialMovingAverage(
        #     mnist_train.MOVING_AVERAGE_DECAY)
        # variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver() #variables_to_restore)


        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(
                const.MODEL_DIR)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path\
                                    .split("/")[-1].split("-")[-1]
                accuracy_score = sess.run(accuracy,
                                            feed_dict=validate_feed)
                print("After %s training step(s), validation "
                        "accuracy = %g" % (global_step, accuracy_score))
                logger.info("After %s training step(s), validation "
                    "accuracy = %g" % (global_step, accuracy_score))
            else:
                print("No checkpoint file found")
                return
    return accuracy_score


def main(argv=None):
    logger = process.get_logger()
    logger.info("Evaluation started")
    evaluate()
    logger.info("Evaluation ended")


if __name__ == "__main__":
    tf.app.run()
