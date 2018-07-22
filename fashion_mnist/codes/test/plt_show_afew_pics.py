# to display a few images from database
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np 


IMAGE_SIZE = 28
DATABASE_PATH = "./fashion_mnist/fashion-mnist/data/fashion"


mnist = input_data.read_data_sets(DATABASE_PATH)

image = mnist.train.images[1]
reshaped_img = np.reshape(
    image,
    [IMAGE_SIZE,
    IMAGE_SIZE]
) * 255
print(reshaped_img)
plt.imshow(reshaped_img)
plt.show()
