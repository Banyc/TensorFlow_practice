# print a few couples of labels from the fashion_mnist database
# purpose: to identify the size and form of labels of fashion_mnist and to determine the placeholder size of variable, y_-input, in train.py 

import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

IMAGE_SIZE = 28
DATABASE_PATH = "./fashion_mnist/fashion-mnist/data/fashion"

mnist = input_data.read_data_sets(DATABASE_PATH)

label = mnist.train.labels[0]
print(label)
