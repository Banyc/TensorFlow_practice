import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np 

import inference
import train


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
        tf.float32,
        [None, inference.NUM_LABELS],
        "y_-input"
    )

    y = inference.inference(x, False, None)

    accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.argmax(y, 1),
                tf.argmax(y_, 1)
            ),
            tf.float32
        )
    )

    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(
            train.MODEL_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        xs = mnist.validation.images
        reshaped_xs = np.reshape(
            xs,
            (-1,
            inference.IMAGE_SIZE,
            inference.IMAGE_SIZE,
            inference.NUM_CHANNELS)
        )
        validate_acc = sess.run(accuracy, {x: reshaped_xs, y_: mnist.validation.labels})
        print(validate_acc)


def main(argv=None):
    mnist = input_data.read_data_sets("./mnist", one_hot=True)
    evaluate(mnist)
    return 0


if __name__ == "__main__":
    main()
