# -*- coding: utf-8 -*-
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np

import const, process, inference


def evaluate():
    logger = process.get_logger()

    with tf.Graph().as_default() as g:

        x = tf.placeholder(
            tf.int32,
            [None, None],
            name="x-input"
        )
        
        y = inference.inference(x, process.get_lex_len(), None)

        tf.argmax(y, 1)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(
                const.MODEL_DIR)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path\
                                    .split("/")[-1].split("-")[-1]
                feed_x = process.get_vec_from_text(const.PREDICT_TEXT)
                # feed_x = feed_x.any()
                y = sess.run(y, feed_dict={x: feed_x})
                print("After %s training step(s), text: '%s', "
                        "prediction: %s" % (global_step, const.PREDICT_TEXT, str(y)))
                logger.info("After %s training step(s), text: '%s', "
                    "prediction = %s" % (global_step, const.PREDICT_TEXT, str(y)))
            else:
                print("No checkpoint file found")
                return


def main(argv=None):
    logger = process.get_logger()
    logger.info("Predict started")
    evaluate()
    logger.info("Predict ended")


if __name__ == "__main__":
    tf.app.run()
