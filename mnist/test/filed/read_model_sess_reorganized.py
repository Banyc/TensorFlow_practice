import tensorflow as tf
from PIL import Image
import numpy as np

MODEL_PATH = "./saved_variables/mnist5.2.1.ckpt"
PIC_PATH = "./pictures/4.png"

INPUT_NODE = 784  # 输入节点
OUTPUT_NODE = 10  # 输出节点

LAYER1_NODE = 500  # 隐藏层神经元数


def inference(input_tensor, weights1, biases1, weights2, biases2):
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
    layer2 = tf.matmul(layer1, weights2) + biases2
    return layer2
    

def Read_vars(sess, path):
    weights1 = tf.Variable(
        tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    weights2 = tf.Variable(
        tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    saver = tf.train.Saver()

    saver.restore(sess, path)
        
    return (weights1, biases1, weights2, biases2)


def Print_var(sess, var1):
    # with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        print(var1.eval(session=sess))
        print()


def Import_pic(path):
    image1 = Image.open(path)
    # 读取顺序 是 逐行读取
    image1_pixels = []
    for j in range(28):
        for i in range(28):
            pixel = image1.getpixel((i, j))
            pixel = 1.0 - pixel  # 背景色与前景色调换
            image1_pixels.append(pixel)
    return image1_pixels


def Guess(sess, image1_pixels, weights1, biases1, weights2, biases2):
    x_ = tf.placeholder(tf.float32, [None, INPUT_NODE], name="x_-input")
    y = inference(x_, weights1, biases1, weights2, biases2)


    # https://blog.csdn.net/wuguangbin1230/article/details/71170903
    image_feed = np.reshape(image1_pixels, (1, -1))  # 转化为 横向量
    result_y = sess.run(y, feed_dict={x_: image_feed})
    print(result_y)
    prediction = sess.run(tf.argmax(result_y, 1))
    print(prediction)


def main(argv=None):
    tf.reset_default_graph()
    sess = tf.Session()

    weights1, biases1, weights2, biases2 = Read_vars(sess, MODEL_PATH)
    # Print_var(sess, weights1)
    image1_pixels = Import_pic(PIC_PATH)
    Guess(sess, image1_pixels, weights1, biases1, weights2, biases2)
    return 0 


if __name__ == "__main__":
    main()

        

