import tensorflow as tf
import numpy as np
from PIL import Image

INPUT_NODE = 784  # 输入节点
OUTPUT_NODE = 10  # 输出节点


LAYER1_NODE = 500  # 隐藏层神经元数

def inference(input_tensor, avg_class, weights1, biases1,
              weights2, biases2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        pass


def read_image():
    image1 = Image.open("1.png")
    image1_pixels = []
    # row_pixels = []
    for i in range(28):
        for j in range(28):
            pixel = image1.getpixel((i, j))
            pixel = 1.0 - pixel  # 背景色与前景色调换
            # row_pixels.append(pixel)
            image1_pixels.append(pixel)
        # image1_pixels.append(row_pixels)
        # row_pixels = []
    return image1_pixels

    # image = tf.gfile.FastGFile("1.png", "r")
    # return image


def verify_an_image(pixels):
    global weights1
    global biases1
    global weights2
    global biases2
    x_ = tf.placeholder(tf.float32, [None, INPUT_NODE], name="x_-input")
    y = inference(x_, None, weights1, biases1, weights2, biases2)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # https://blog.csdn.net/wuguangbin1230/article/details/71170903
        image_feed = np.reshape(pixels, (1, -1))
        result_y = sess.run(y, feed_dict={x_: image_feed})
        print(result_y)
        prediction = sess.run(tf.argmax(result_y, 1))
        print(prediction)


weights1 = tf.Variable(
    tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))  # 为什么是这个矩阵?
biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))  # 纵向量

weights2 = tf.Variable(
    tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))  # 标准差
biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

# 创建saver 对象
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "saved variables/mnist5.2.1.ckpt")

pixels = read_image()
verify_an_image(pixels)
