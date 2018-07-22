import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST")
image1 = mnist.train.images[0]
print(image1)



