import tensorflow as tf 
import const, inference, train, process
import matplotlib.pyplot as plt
import numpy as np

IMAGE_ORDER = 5


'''
read only one testing datum
'''
def get_datum(order):
    fp = open(const.TEST_PATH)
    for i in range(order):
        fp.readline()
    test_x = []
    test_y = []
    line = fp.readline()
    if line:
        print(line)
        value = line.split(',')
        test_x.append(value[0])
        test_y.append(value[1])
    test_datum = list(zip(test_x, test_y))
    print(test_datum)
    return (test_datum)


def predict():
    x = tf.placeholder(
        tf.float32, 
        [None, 
        const.IMAGE_HEIGHT,
        const.IMAGE_WIDTH,
        const.NUM_CHANNELS],
        "predict_x-input")
    y_ = tf.placeholder(tf.float32, [None, const.MAX_CAPTCHA * const.CHAR_SET_LEN], "predict_y_-input")
    global_step = tf.Variable(0, False)

    y = inference.inference(x, False, None)

    reshaped_y = tf.reshape(y, [-1, const.MAX_CAPTCHA, const.CHAR_SET_LEN])

    max_idx_p = tf.argmax(reshaped_y, 2)
    max_idx_l = tf.argmax(tf.reshape(y_, [-1, const.MAX_CAPTCHA, const.CHAR_SET_LEN]), 2)

    # reshaped_y = tf.reshape(y, [-1, const.MAX_CAPTCHA * const.CHAR_SET_LEN])
    



    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer()
        ckpt = tf.train.get_checkpoint_state(const.MODEL_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            sess.close()
            return
        
        test_x, test_y, _ = process.get_next_batch(get_datum(IMAGE_ORDER), 0, 1)
        plt.imshow(np.reshape(test_x, [const.IMAGE_HEIGHT, const.IMAGE_WIDTH]))
        plt.show()

        print(sess.run(reshaped_y, {x: test_x, y_: test_y}))
        print(sess.run(max_idx_p, {x: test_x, y_: test_y}))
        print(sess.run(max_idx_l, {x: test_x, y_: test_y}))
        text_y = process.vec2text(sess.run(reshaped_y, {x: test_x, y_: test_y}))

        print(text_y)
        # print(sess.run(text_y, {x: test_x, y_: test_y}))


def main(argv=None):
    predict()


if __name__ == "__main__":
    tf.app.run()
