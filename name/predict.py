# -*- utf-8 -*-
import tensorflow as tf 
import numpy as np
import const, model, process


def predict(name_list):
    xs, input_size, voc_len = process.get_word_vec(name_list)

    x = tf.placeholder(
        tf.int32,
        [None,
        input_size],
        "input-x"  
    )
    dropout_keep_prob = tf.placeholder(tf.float32)

    
 
    y = model.model(x, input_size, voc_len, dropout_keep_prob)

    predictions = tf.argmax(y, 1)
 
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 恢复前一次训练
        ckpt = tf.train.get_checkpoint_state(const.MODEL_DIR)
        if ckpt != None:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("没找到模型")
 
        
        
        res = sess.run(predictions, {x: xs, dropout_keep_prob: 1.0})
 
        i = 0
        for name in name_list:
            print(name, '女' if res[i] == 0 else '男')
            i += 1
 

predict(["白富美", "高帅富", "王婷婷", "田野"])
