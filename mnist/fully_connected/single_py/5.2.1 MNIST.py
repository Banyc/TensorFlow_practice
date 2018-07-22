import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784  # 输入节点
OUTPUT_NODE = 10  # 输出节点


LAYER1_NODE = 500  # 隐藏层神经元数

BATCH_SIZE = 100  # batch的大小

LEARNING_RATE_BASE = 0.8  # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习率衰减
REGULARIZATION_RATE = 0.0001  # 正则化率
TRAINING_STEPS = 30000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率?


def inference(input_tensor, avg_class, weights1, biases1,
              weights2, biases2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:  # 滑动平均衰减
        layer1 = tf.nn.relu(
            tf.matmul(input_tensor, avg_class.average(weights1)) +
            avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + \
                avg_class.average(biases2)


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y-input")  # 正确答案

    weights1 = tf.Variable(
        tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))  # 为什么是这个矩阵? 考虑与其相乘的tensor's shape
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))  # 向量 没有规定横纵

    weights2 = tf.Variable(
        tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))  # 标准差
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    y = inference(x, None, weights1, biases1,
                  weights2, biases2)

    global_step = tf.Variable(0, trainable=False)  # 记录训练轮数 是不可训练的

    # 以下讲的是 滑动平均衰减
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)

    variable_averages_op = variable_averages.apply(
        tf.trainable_variables())

    average_y = inference(
        x, variable_averages, weights1, biases1, weights2, biases2)

    # 以下交叉熵 定义 loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))  # 把(一个长度为10的数组内)正确答案 中正确的那一个数字（就是概率为1的那个数字）取出来，这就是对应的编号了
    cross_entropy_mean = tf.reduce_mean(cross_entropy)  # 获得 交叉熵 的 所有元素 的 平均值（一个数字）

    # 以下计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)  # 实例化
    regularization = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularization  # 为什么是loss函数的平均数(交叉熵损失)？
    # 以下学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,  # 当前迭代轮数
        mnist.train.num_examples / BATCH_SIZE,  # 总迭代次数
        LEARNING_RATE_DECAY)

    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
                    .minimize(loss, global_step=global_step)

    # 反向传播（此处是确保反向传播已经进行（train_step 和 variable_averages_op 都已经发生改变））
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")  # Name一定要写！ 不明白这个是什么意思

    # 检验答案
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))  # Return boolean
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 求正确率：把bool转化为int然后求平均值

    saver = tf.train.Saver()  # 创建saver对象，保存模型

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 准备验证数据 用来判断停止的条件和训练效果 是 训练数据
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}

        # 准备测试数据
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # 训练 训练数据
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("%d\t%g" % (i, validate_acc))

            xs, ys = mnist.train.next_batch(BATCH_SIZE)  # 什么格式? List
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("Finally, %d\t%g", TRAINING_STEPS, test_acc)

        saver.save(sess, "saved_variables/mnist5.2.1.ckpt")  # 保存模型


def main(argv=None):
    mnist = input_data.read_data_sets("MNIST", one_hot=True)
    train(mnist)


if __name__ == "__main__":
    tf.app.run()
