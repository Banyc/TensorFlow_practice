# To predict the label of a given image with a single channel
import tensorflow as tf 
import numpy as np 
import os
import matplotlib.pyplot as plt 

IMAGE_SIZE = 28
NUM_CHANNELS = 1

# here paths the image
IMAGE_PATH = "./fashion_mnist/images/"
IMAGE_NAME = "2.png"

MODEL_PATH = "./fashion_mnist/codes/LeNet-5/model"

# Importing files from different folder
# source: https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import sys
sys.path.insert(0, './fashion_mnist/codes/LeNet-5/')

import inference


def read_pic():
    image_raw_data = tf.gfile.FastGFile(os.path.join(IMAGE_PATH, IMAGE_NAME), 'rb').read()
    img_data = tf.image.decode_png(image_raw_data)
    img_data = tf.image.resize_images(
        img_data, [IMAGE_SIZE, IMAGE_SIZE]
    )
    img_data = tf.reshape(img_data,
        [-1, 
        IMAGE_SIZE,
        IMAGE_SIZE,
        NUM_CHANNELS])
    return img_data


def predict(labels):
    x = tf.placeholder(
        tf.float32,
        [None,
        IMAGE_SIZE,
        IMAGE_SIZE,
        NUM_CHANNELS]
    )

    y = inference.inference(x, False, None)

    prediction_index = tf.argmax(y, 1)[0]
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(
            MODEL_PATH
        )
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(
                sess, ckpt.model_checkpoint_path
            )
        img_data = read_pic().eval()
        # test input image
        print(np.shape(img_data))
        plt.imshow(np.reshape(img_data,
            [IMAGE_SIZE, IMAGE_SIZE]))
        plt.show()
        # end test

        prediction_index_value = sess.run(prediction_index, {x: img_data})
        print(labels[prediction_index_value])


def main(argv=None):
    labels = ['t_shirt_top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']
    predict(labels)
    return 0


if __name__ == "__main__":
    tf.app.run()
