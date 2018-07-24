# 验证 模型 正确率，区别于 train.py
import tensorflow as tf 
import const, train, process, inference


def evaluate(test_xs, test_ys):
    x = tf.placeholder(
        tf.float32,
        [None,
        const.IMAGE_HEIGHT,
        const.IMAGE_WIDTH,
        const.NUM_CHANNELS],
        "x-input"
    )
    y_ = tf.placeholder(
        tf.float32,
        [None, inference.NUM_LABELS],
        "y_-input"
    )
    global_step = tf.Variable(0, False)
    
    y = inference.inference(x, False, None)

    reshaped_y = tf.reshape(
        y, 
        [-1,
        const.MAX_CAPTCHA,
        const.CHAR_SET_LEN]
    )
    reshaped_y_ = tf.reshape(
        y_, 
        [-1,
        const.MAX_CAPTCHA,
        const.CHAR_SET_LEN]
    )
    max_idx_p = tf.argmax(reshaped_y, 2)
    max_idx_l = tf.argmax(reshaped_y_, 2)

    accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(max_idx_l, max_idx_p),
            tf.float32
        )
    )
    
    saver = tf.train.Saver()
    with tf.Session() as sess:

        tf.global_variables_initializer()
        ckpt = tf.train.get_checkpoint_state(const.MODEL_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            # global_step = int(ckpt.model_checkpoint_path.split('.')[-1].split('-')[-1])
        else:
            print('No checkpoint file found')
            sess.close()
            return

        acc_value = sess.run(accuracy, {x: test_xs, y_: test_ys})

        return (global_step.eval(), acc_value)


def main(argv=None):
    test_data = process.get_test_dataset()
    test_xs, test_ys, _ = process.get_next_batch(test_data, 0, len(test_data))
    
    global_step, acc_value = evaluate(test_xs, test_ys)
    print("Global step: #%d, accuracy: %g" % (global_step, acc_value))
    logger = process.get_logger()
    logger.info("Global step: #%d, accuracy: %g" % (global_step, acc_value))
    return 0
        
            
if __name__ == "__main__":
    tf.app.run()
