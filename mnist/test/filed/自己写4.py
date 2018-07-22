import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
LAYER1_NODE = 500
LAYER2_NODE = 10

TRAIN_STEPS = 2000
BATCH_SIZE = 100

REGULARIZATION_RATE = 0.0001

def inference(input_data, weights1, biases1, weights2, biases2):
    layer1 = tf.nn.relu(tf.matmul(input_data, weights1) + biases1)
    layer2 = tf.matmul(layer1, weights2) + biases2
    return layer2


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], "x_input")
    y_ = tf.placeholder(tf.float32, [None, LAYER2_NODE], "y_-input")

    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, tf.float32, [LAYER1_NODE]))
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,LAYER2_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, tf.float32, [LAYER2_NODE]))

    y = inference(x, weights1, biases1, weights2, biases2)

    # return 矩阵; labels 是 标准答案，logits 是 预测概率，交叉熵越小 越接近，越好
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
    # return 常数; 计算 batch 的所有 交叉熵 的 平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 这个是一个 函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)

    loss = cross_entropy_mean + regularization

    train_step = tf.train.AdamOptimizer().minimize(loss)
    
    with tf.control_dependencies([train_step]):
        train_op = tf.no_op(name="train")

    accuracy = tf.reduce_mean(
        tf.cast(  # int -> float32 方便获得小数
            tf.equal(  # return 矩阵
                tf.argmax(y_, 1),   # 正确 label
                tf.argmax(y, 1)  # 向前传递 结果 
            ),
            tf.float32
        )
    )

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        for i in range(TRAIN_STEPS):
            if i % 1000 == 0:
                acc = sess.run(accuracy, validate_feed)
                print("After %d steps, accuracy is %lf" % (i, acc))
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, {x: xs, y_: ys})
        test_acc = sess.run(accuracy, test_feed)
        print("Final test with acc: %lf" % (test_acc))
        

def main(argv=None):
    mnist = input_data.read_data_sets("./mnist", one_hot=True)
    train(mnist)
    return 0


if __name__ == "__main__":
    tf.app.run()
