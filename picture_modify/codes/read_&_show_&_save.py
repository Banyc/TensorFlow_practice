# read an local image, display it with MatPlotLib and save the decoded image
import tensorflow as tf 
import matplotlib.pyplot as plt 
import os

IMAGE_PATH = "./picture_modify/pics"
INPUT_IMAGE_NAME = "cat.jpg"
OUTPUT_IMAGE_NAME = "output_cat.jpg"

image_raw_data = tf.gfile.FastGFile(os.path.join(IMAGE_PATH, INPUT_IMAGE_NAME), 'rb').read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)

    print(img_data.eval())

    plt.imshow(img_data.eval())
    plt.show()


    encode_image = tf.image.encode_jpeg(img_data)
    # the same as "f = tf.gfile.GFile()" and "f.close()"
    # Ques: difference in file size
    with tf.gfile.GFile(os.path.join(IMAGE_PATH, OUTPUT_IMAGE_NAME), "wb") as f:
        f.write(encode_image.eval())
