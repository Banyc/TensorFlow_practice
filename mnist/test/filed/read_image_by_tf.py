import tensorflow as tf
# https://blog.csdn.net/grllery/article/details/78471970

image_raw = tf.gfile.FastGFile("1.png", "rb").read()

with tf.Session() as sess:
    image_data = tf.image.decode_png(image_raw)
    # image_data = tf.cast(image_data, tf.float32)
    # image_data = tf.image.convert_image_dtype(image_data, dtype=tf.uint8)
    image_data = tf.image.rgb_to_grayscale(image_data)
    result = sess.run(image_data)
    print(result)

