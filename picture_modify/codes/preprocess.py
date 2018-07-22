# preprecess images before training
# code cited from page #182
import tensorflow as tf 

import numpy as np 
import matplotlib.pyplot as plt

import os

IMAGE_PATH = "./picture_modify/pics"
INPUT_IMAGE_NAME = "cat.jpg"
OUTPUT_IMAGE_NAME = "output_cat.jpg"

def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        # 饱和度
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        # 色相
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering == 2:
        pass
    # tf.clip_by_value(A, min, max)：输入一个张量A，把 A 中的每一个元素的值都压缩在 min 和 max 之间。小于 min 的让它等于 min，大于 max 的元素的值等于max。
    # source: https://blog.csdn.net/UESTC_C2_403/article/details/72190248
    return tf.clip_by_value(image, 0.0, 1.0)


def preprocess_for_train(image, height, width, bbox):
    if bbox == None:
        # y_min, x_min, y_max, x_max
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
            dtype=tf.float32, shape=[1, 1, 4])  # 无法理解 维度
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(
            image, dtype=tf.float32
        )
    
    # 以下修改图片 (一共 4 步)
    # 以下随机生成 新的 bbox
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image), bounding_boxes=bbox
    )
    # 图片随机 切片 严格按照 新的 bbox
    distorted_image = tf.slice(image, bbox_begin, bbox_size)
    # 图片 改变size 符合 input_sector 的 size
    distorted_image = tf.image.resize_images(
        distorted_image, [height, width], method=np.random.randint(4)
    )
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = distort_color(distorted_image, np.random.randint(2))
    return distorted_image


# no difference between .FastGFile & .GFile
# source: https://github.com/tensorflow/tensorflow/issues/12663
image_raw_data = tf.gfile.FastGFile(os.path.join(IMAGE_PATH, INPUT_IMAGE_NAME), "rb").read()
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])

    for i in range(6):
        result = preprocess_for_train(img_data, 299, 299, boxes)
        plt.imshow(result.eval())
        plt.show()
