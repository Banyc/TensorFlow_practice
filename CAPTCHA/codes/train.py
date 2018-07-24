# ! #/usr/bin/env python
# -*- coding=utf-8 -*-
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import inference, const, process, evaluate


# 训练#####
def train(train_data, test_data):
	# x = tf.placeholder(tf.float32, [None, const.IMAGE_HEIGHT * const.IMAGE_WIDTH], "x_input")
	x = tf.placeholder(
		tf.float32,
		[None, 
		const.IMAGE_HEIGHT,
		const.IMAGE_WIDTH,
		const.NUM_CHANNELS],
		"x-input"
	)
	y_ = tf.placeholder(tf.float32, [None, const.MAX_CAPTCHA * const.CHAR_SET_LEN], "y_-input")
	global_step = tf.Variable(0, False)
	# keep_prob = tf.placeholder(tf.float32) # dropout
	regularizer = tf.contrib.layers.l2_regularizer(const.REGULARIZATION_RATE)
	# TODO 好像有问题
	# y = inference.inference(x, True, regularizer)
	y = inference.inference(x, True, None)
	
	# loss
	#之前tensorflow没升级的时候，用的是targets，当升级到1.3时，用的是labels
	# cross_entropy_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1)))
	cross_entropy_mean = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_))
    # 最后一层用来分类的softmax和sigmoid有什么不同？
	
	loss = cross_entropy_mean #+ tf.add_n(tf.get_collection('losses'))
	
	learning_rate = tf.train.exponential_decay(
		const.LEARNING_RATE_BASE,
		global_step,
		const.TRAINING_STEPS / const.BATCH_SIZE,
		const.DECAY_LEARNING_RATE
	)
	# 优化器，optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
	# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, global_step)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step)

	# == end ==
 
	predict = tf.reshape(y, [-1, const.MAX_CAPTCHA, const.CHAR_SET_LEN])
	
    #函数会返回tensor中参数指定的维度中的最大值的索引
    #最后输出的是一个，4*26维度的向量,所以最大的索引值为4个，和正确的结果进行对比
	max_idx_p = tf.argmax(predict, 2)
	max_idx_l = tf.argmax(tf.reshape(y_, [-1, const.MAX_CAPTCHA, const.CHAR_SET_LEN]), 2)
	
    #通过tf.equal方法可以比较预测结果与实际结果是否相等：
	correct_pred = tf.equal(max_idx_p, max_idx_l)
	
    #这行代码返回一个布尔列表。为得到哪些预测是正确的，我们可用如下代码将布尔值转换成浮点数：
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	
	# global train_data
	#Tensorflow针对这一需求提供了Saver类，保存模型，恢复模型变量。
	saver = tf.train.Saver()
 
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		ckpt = tf.train.get_checkpoint_state(const.MODEL_DIR)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
		
		logger = process.get_logger()
	
		cursor = 0
		for i in range(const.TRAINING_STEPS):
			xs, ys, cursor = process.get_next_batch(train_data, cursor)
			sess.run(optimizer, {x: xs, y_: ys})
			if i % 50 == 0:
				saver.save(sess, os.path.join(const.MODEL_DIR, const.MODEL_NAME))

				loss_value, cross_entropy_mean_value = sess.run((loss, cross_entropy_mean), {x: xs, y_: ys})
				
				# test_xs, test_ys, _ = process.get_next_batch(test_data, 0, len(test_data))

				# # acc_value = sess.run(accuracy, {x: test_xs, y_: test_ys})
				# acc_value, global_step_value = evalidate.evalidate(test_xs, test_ys)
				
				# logger.info("Global step: #%d, loss: %g, accuracy: %g" % (global_step.eval(), loss_value, acc_value))
				# logger.info("Global step: #%d, loss: %g, accuracy: %g" % (global_step_value, loss_value, acc_value))
				logger.info("Global step: #%d, loss: %g, cross_entropy_mean: %g" % (global_step.eval(), loss_value, cross_entropy_mean_value))

		saver.save(sess, os.path.join(const.MODEL_DIR, const.MODEL_NAME), global_step)

		loss_value, cross_entropy_mean_value = sess.run((loss, cross_entropy_mean), {x: xs, y_: ys})
				
		# test_xs, test_ys, _ = process.get_next_batch(test_data, 0, len(test_data))

		# # acc_value = sess.run(accuracy, {x: test_xs, y_: test_ys})
		# acc_value, global_step_value = evalidate.evalidate(test_xs, test_ys)

		# logger.info("Global step: #%d (final step), loss: %g, accuracy: %g" % (global_step.eval(), loss_value, acc_value))
		logger.info("Global step: #%d (final step), loss: %g, cross_entropy_mean: %g" % (global_step.eval(), loss_value, cross_entropy_mean_value))

	
def main(argv=None):
	logger = process.get_logger()
	logger.info("Training started.")
	train_data = process.get_train_dataset()
	test_data = process.get_test_dataset()
	train(train_data, test_data)
	logger.info("Training ended.")
	

if __name__ == "__main__":
	tf.app.run()
	
'''
 
使用Saver.save()方法保存模型：
sess：表示当前会话，当前会话记录了当前的变量值
checkpoint_dir + 'model.ckpt'：表示存储的文件名
global_step：表示当前是第几步
'''
