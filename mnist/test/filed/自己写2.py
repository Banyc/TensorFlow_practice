import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


INPUT_NODE = 784
LAYER1_NODE = 500
OUTPUT_NODE = 10

TRAINING_STEPS = 2000
BATCH_SIZE = 100


def inference(input_tensor, weight1, biases1, weight2, biases2):
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weight1) + biases1)
    layer2 = tf.matmul(layer1, weight2) + biases2
    return layer2


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], "x-input")
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], "y-input")

    weight1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, tf.float32, [LAYER1_NODE]))
    weight2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, tf.float32, [OUTPUT_NODE]))

    y = inference(x, weight1, biases1, weight2, biases2)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)  #
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean

    train_step = tf.train.AdamOptimizer().minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  #
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))  #

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
                validate_acc = sess.run(accuracy, validate_feed)
                print(i, validate_acc)

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            train_feed = {x: xs, y_: ys}
            sess.run(train_step, train_feed)

        test_feed = {x: mnist.test.images, y_: mnist.test.labels}
        test_acc = sess.run(accuracy, test_feed)
        print(TRAINING_STEPS, test_acc)


mnist = input_data.read_data_sets("MNIST", one_hot=True)
train(mnist)

