import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


INPUT_NODE = 784
LAYER1_NODE = 500
OUTPUT_NODE = 10

BATCH_SIZE = 500
TRAINING_STEPS = 2000



def inference(input_tensor, weights1, biases1, weights2, biases2):
    layer1 = tf.nn.relu(tf.matmul(weights1, input_tensor) + biases1)
    layer2 = tf.matmul(weights2, layer1) + biases2
    return layer2


def train(mnist):
    x = tf.placeholder(tf.float32, [INPUT_NODE, None], "x-input")
    y_ = tf.placeholder(tf.float32, [OUTPUT_NODE, None], "y_-input")
    # x = tf.placeholder(tf.float32, [None, INPUT_NODE], "x-input")
    # y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], "y_-input")
    # x = tf.transpose(x)
    # y_ = tf.transpose(y_)
    
    weights1 = tf.Variable(tf.truncated_normal([LAYER1_NODE, INPUT_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, tf.float32, [LAYER1_NODE]))
    # biases1 = tf.transpose(tf.Variable(tf.constant(0.1, tf.float32, [LAYER1_NODE])))
    weights2 = tf.Variable(tf.truncated_normal([OUTPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, tf.float32, [OUTPUT_NODE]))
    # biases2 = tf.transpose(tf.Variable(tf.constant(0.1, tf.float32, [OUTPUT_NODE])))

    y = inference(x, weights1, biases1, weights2, biases2)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean

    train_step = tf.train.AdamOptimizer().minimize(loss)

    with tf.control_dependencies([train_step]):
        train_op = tf.no_op(name="train_op")
    
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        test_feed = {x: tf.transpose(mnist.test.images), y_: tf.transpose(mnist.test.labels)}
        validate_feed = {x: tf.transpose(mnist.validation.images), y_: tf.transpose(mnist.validation.labels)}
        
        for i in range(TRAINING_STEPS):
            
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d steps, the accuracy is %g" % (i, validate_acc))

            xs, y_s = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: tf.transpose(xs), y_: tf.transpose(y_s)})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("The final test in step %d accuracy is %g" % (TRAINING_STEPS, test_acc))


def main(argv=None):
    mnist = input_data.read_data_sets("./MNIST", one_hot=True)
    train(mnist)


if __name__ == "__main__":
    main()
        








    



