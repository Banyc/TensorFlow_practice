# predict its label from an trimmed image and show
# the testing image is from the database of fashion_mnist, not the handwritten one

import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

import train, inference


def predict(mnist, labels):
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
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(
            train.MODEL_PATH
        )
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(
                sess, ckpt.model_checkpoint_path)
        valid_num = np.random.randint(mnist.validation.num_examples)
        validate_image = mnist.validation.images[valid_num]
        validate_label = mnist.validation.labels[valid_num]
        reshaped_image = np.reshape(
            validate_image,
            [-1,
            inference.IMAGE_SIZE,
            inference.IMAGE_SIZE,
            inference.NUM_CHANNELS]
        ) 
        reshaped_label = np.reshape(
            validate_label,
            [-1]
        )
        prediction_index = sess.run(tf.argmax(y, 1), 
            {x: reshaped_image, y_: reshaped_label})

        print("correct answer: \t%s" % (labels[reshaped_label[0]]))
        print("prediction: \t%s" % (labels[prediction_index[0]]))

        image_forShow = np.reshape(
            validate_image,
            [inference.IMAGE_SIZE,
            inference.IMAGE_SIZE]
        )
        plt.imshow(image_forShow)
        plt.show()


def main(argv=None):
    labels = ['t_shirt_top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']
    mnist = input_data.read_data_sets(
        train.DATABASE_PATH
    )
    predict(mnist, labels)
    return 0


if __name__ == "__main__":
    tf.app.run()
