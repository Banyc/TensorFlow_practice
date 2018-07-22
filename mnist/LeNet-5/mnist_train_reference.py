# A reference given from Internet
should not be regarded as the 
# https://github.com/cookeem/TensorFlow_learning_notes/blob/master/Chapter06/LeNet-5/LeNet5_train.ipynb
# In [1]:
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import os
import numpy as np
# 1. 定义神经网络相关的参数
# In [2]:
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 6000
MOVING_AVERAGE_DECAY = 0.99
# 2. 定义训练过程
# In [ ]:
def train(mnist):
    # 定义输出为4维矩阵的placeholder
    x = tf.placeholder(tf.float32, [
            BATCH_SIZE,
            mnist_inference.IMAGE_SIZE,
            mnist_inference.IMAGE_SIZE,
            mnist_inference.NUM_CHANNELS],
        name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
    
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x,False,regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
        
    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE,
                mnist_inference.IMAGE_SIZE,
                mnist_inference.IMAGE_SIZE,
                mnist_inference.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})

            if i % 50 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
# 3. 主程序入口
# In [ ]:
def main(argv=None):
    # mnist = input_data.read_data_sets("../../datasets/MNIST_data", one_hot=True)
    mnist = input_data.read_data_sets("./MNIST", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    main()
# Extracting ../../datasets/MNIST_data/train-images-idx3-ubyte.gz
# Extracting ../../datasets/MNIST_data/train-labels-idx1-ubyte.gz
# Extracting ../../datasets/MNIST_data/t10k-images-idx3-ubyte.gz
# Extracting ../../datasets/MNIST_data/t10k-labels-idx1-ubyte.gz
# After 1 training step(s), loss on training batch is 4.26947.
# After 101 training step(s), loss on training batch is 1.01421.
# After 201 training step(s), loss on training batch is 0.928992.
# After 301 training step(s), loss on training batch is 0.810302.