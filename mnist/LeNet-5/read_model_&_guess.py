# to read model and to guess input images given out of database of mnist
# written by Banic
# partly referred from mnist_eval
import tensorflow as tf
import numpy as np
import os
from PIL import Image

import mnist_inference
import mnist_train

PICS_PATH = "./pictures"
PIC_NAME = "7.png"


def read_pic():
    image_raw = Image.open(os.path.join(PICS_PATH, PIC_NAME))
    image = []
    # pixel =
    for y in range(mnist_inference.IMAGE_SIZE):
        for x in range(mnist_inference.IMAGE_SIZE):
            pixel = image_raw.getpixel((x, y))
            pixel = 1.0 - pixel
            image.append(pixel)
    image = np.reshape(image, 
        (-1,
        mnist_inference.IMAGE_SIZE,
        mnist_inference.IMAGE_SIZE,
        mnist_inference.NUM_CHANNELS))
    return image


def guess(image):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32,
            [None,
            mnist_inference.IMAGE_SIZE,
            mnist_inference.IMAGE_SIZE,
            mnist_inference.NUM_CHANNELS],
            "x-input")
        # y_ = tf.placeholder(tf.float32,
        #     [None, mnist_inference.NUM_LABELS]
        # )
        y = mnist_inference.inference(x, False, None)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(
                mnist_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            predict_y_raw = sess.run(y, {x: image})
            print(predict_y_raw)
            predict_y = sess.run(tf.argmax(predict_y_raw, 1))
            print(predict_y)


def main(argv=None):
    image = read_pic()
    guess(image)


if __name__ == "__main__":
    main()
