# Aborted
import tensorflow as tf
import const


# 定义CNN
def crack_captcha_cnn(input_tensor, keep_prob, w_alpha=0.01, b_alpha=0.1):
	x = tf.reshape(input_tensor, shape=[-1, const.IMAGE_HEIGHT, const.IMAGE_WIDTH, 1])
 
	#w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #
	#w_c2_alpha = np.sqrt(2.0/(3*3*32)) 
	#w_c3_alpha = np.sqrt(2.0/(3*3*64)) 
	#w_d1_alpha = np.sqrt(2.0/(8*32*64))
	#out_alpha = np.sqrt(2.0/1024)
 
	# 3 conv layer
	print("xxxx")
	print(input_tensor.get_shape())   # 图片大小 64*128 = 8192
	print(x.get_shape())   #64 * 64 * 128 * 1
	
	#layer_1
	w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 32]))
	b_c1 = tf.Variable(b_alpha*tf.random_normal([32]))
	
	#tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)，参数属性
	#x[64,64,128,1]  w_c1[3,3,1,32]
	# 卷积后：32个feature，每个shape还是（？，64，128，32）
	conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
	
	print(conv1.get_shape()) 
	#max_pool之后，就变成了，(?, 32, 64, 32)
	#max_pool不同于卷积，它能保证位置不变性，降低计算参数
	conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	print(conv1.get_shape()) 
	
	conv1 = tf.nn.dropout(conv1, keep_prob)
	print(conv1.get_shape()) 
 
	#layer_2
	w_c2 = tf.Variable(w_alpha*tf.random_normal([3, 3, 32, 64]))
	b_c2 = tf.Variable(b_alpha*tf.random_normal([64]))
	conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
	#做完conv2d,(?, 32, 64, 64),
	conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	#做完pool之后（？，16, 32, 64)
	conv2 = tf.nn.dropout(conv2, keep_prob)
    
	#layer_3
	w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]))
	b_c3 = tf.Variable(b_alpha*tf.random_normal([64]))
	conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
	#conv2d 后的数据，(?, 16, 32, 64）
	conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	#pool后的维度 (?, 8, 16, 64)
	conv3 = tf.nn.dropout(conv3, keep_prob)
	print(b_c3.get_shape()) 
	
	# Fully connected layer 64*128，全连接层
	w_d = tf.Variable(w_alpha*tf.random_normal([8192, 1024]))
	b_d = tf.Variable(b_alpha*tf.random_normal([1024]))
 
	#reshap（-1,8192），8*16*64 = 8192
	dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
	#dense的shape(?, 8192)，w_b（8192,1024），输出为1024，
	
	print(dense.get_shape())
	dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
	
	#dense = （？，8192）*（8192,0124）+（1024）
	#dense = （？，1024）
	dense = tf.nn.dropout(dense, keep_prob)
	
	#全连接层的输出是1024维度的
    #全连接层后接着一层输出层：1024*（4*26）
	w_out = tf.Variable(w_alpha*tf.random_normal([1024, const.MAX_CAPTCHA * const.CHAR_SET_LEN]))
	b_out = tf.Variable(b_alpha*tf.random_normal([const.MAX_CAPTCHA * const.CHAR_SET_LEN]))
	#dense = (?, 1024)
	#w_out = （1024,104）
	#b_out = （104）
	out = tf.add(tf.matmul(dense, w_out), b_out)
	#out的shape为（？，104）
	
	#out = tf.nn.softmax(out)
	return out