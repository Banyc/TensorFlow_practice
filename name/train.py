import tensorflow as tf 
import numpy as np 
import os
import model, const, process


# 训练
def train_neural_network():
    Train = process.Data_Process()
    input_size = Train.max_name_length
    voc_len = Train.voc_len

    x = tf.placeholder(
        tf.int32,
        [None,
        input_size],
        "input-x"  
    )
    y_ = tf.placeholder(
        tf.float32,
        [None, const.NUM_CLASSES],
        "input-y_"
    )
    dropout_keep_prob = tf.placeholder(tf.float32)
    global_step = tf.Variable(0, False)

    y = model.model(x, input_size, voc_len, dropout_keep_prob)

 
    optimizer = tf.train.AdamOptimizer(1e-3)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step)
 
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(const.MODEL_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
 
        # for i in range(201):
        while Train.cur_epoch < const.EPOCH:
            batch_xs, batch_ys = Train.get_train_batch()
            _, loss_ = sess.run([train_op, loss], feed_dict={x: batch_xs, y_: batch_ys, dropout_keep_prob: 0.5})
            print(global_step.eval(), loss_)
            logger = process.get_logger()
            logger.info("Global_step: %d, loss: %g", global_step.eval(), loss_)
            # 保存模型
            if global_step.eval() % 50 == 0:
                saver.save(sess, os.path.join(const.MODEL_DIR, const.MODEL_NAME), global_step=global_step)
                

train_neural_network()