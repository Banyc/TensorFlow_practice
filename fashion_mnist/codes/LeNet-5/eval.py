import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np 
import os
import datetime

import train, inference

LOG_NAME = "log.txt"


def evaluate(mnist):
    x = tf.placeholder(
        tf.float32,
        [None,
        inference.IMAGE_SIZE,
        inference.IMAGE_SIZE,
        inference.NUM_CHANNELS],
        "x-input"
    )
    y_ = tf.placeholder(
        tf.int64,
        [None],
        "y_-input"
    )
    y = inference.inference(x, False, None)

    accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(
                y_, tf.argmax(y, 1)
            ),
            tf.float32
        )
    )

    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(
            train.MODEL_PATH
        )
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(
                sess,
                ckpt.model_checkpoint_path
            )
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            xs = mnist.validation.images
            reshaped_xs = np.reshape(
                xs,
                [-1,
                inference.IMAGE_SIZE,
                inference.IMAGE_SIZE,
                inference.NUM_CHANNELS]
            )
        validate_acc = sess.run(accuracy, {x: reshaped_xs, y_: mnist.validation.labels})
        print(global_step, validate_acc)
    return (global_step, validate_acc)


def log(global_step, accuracy):
    IsExist = os.path.isfile(os.path.join(train.MODEL_PATH, LOG_NAME))
    fp = open(os.path.join(train.MODEL_PATH, LOG_NAME), "a")
    if not IsExist:
        fp.write("Local_time\tglobal_step\taccuracy\n")
    fp.write(datetime.datetime.now().isoformat() + "\t" + global_step + "\t" + str(accuracy) + '\n')
    fp.close()
    

def main(argv=None):
    mnist = input_data.read_data_sets(train.DATABASE_PATH)
    global_step, accuracy = evaluate(mnist)
    log(global_step, accuracy)
    return 0


if __name__ == "__main__":
    tf.app.run()
