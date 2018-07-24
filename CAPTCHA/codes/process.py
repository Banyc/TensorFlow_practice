import tensorflow as tf 
import numpy as np 
import logging
import random
from PIL import Image
import const


'''
'''
def get_logger():
    FORMAT = '[%(asctime)s, %(levelname)-7s]: %(message)s'

    #日志打印到文件中
    logging.basicConfig(format=FORMAT,filename=const.LOG_PATH,filemode='a')
    logger = logging.getLogger('Train_128')
    logger.setLevel(logging.INFO)
    # logger.setLevel(logging.DEBUG)
    return logger
 
 
# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img
 
 
#每一个返回的vector都是一个shape为4*26维度的向量；
def text2vec(text):
    text = text.strip()
    text_len = len(text) 
    #print(text)
    #print(text_len)
    #print(MAX_CAPTCHA)
    if text_len > const.MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')
 
    vector = np.zeros(const.MAX_CAPTCHA * const.CHAR_SET_LEN)
    def char2pos(c):
        k = ord(c)-97
        return k
    for i, c in enumerate(text):
        idx = i * const.CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector

# 向量转回文本 BUG: output is "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text=[]
    for i, c in enumerate(char_pos):
        char_at_pos = i #c/63
        char_idx = c % const.CHAR_SET_LEN
        char_code = char_idx + 97
        text.append(chr(char_code))
    return "".join(text)
 
"""
#向量（大小MAX_CAPTCHA*CHAR_SET_LEN）用0,1编码每26个编码一个字符，这样位置也有，字符也有
"""


'''
Warning: the returns are [(-1,), (-1,)] in shape and in which x_train is made from all paths!
'''
def get_train_dataset():
    #数据类型，两列，第一列是图片的name，第二列是图片的label（一个长度为4的字符串）

    y_train = []
    x_train = []
    
    ff = open(const.TRAIN_PATH)
    while 1:
        line = ff.readline()
        if not line:
            break
        value = line.split('\n')[0].split(",")
        img = value[0]
        label = value[1]
        x_train.append(img)
        y_train.append(label)
    ff.close()
    train_data = list(zip(x_train, y_train))
    random.shuffle(train_data)

    max_train = len(train_data)
    # print(max_train)
    return (train_data)


'''
Warning: the returns are [(-1,), (-1,)] in shape and in which x_train is made from all paths!
'''
def get_test_dataset():
    fff = open(const.TEST_PATH)
    x_test=[]
    y_test=[]
    while 1:
        line = fff.readline()
        if not line:
            break
        value = line.split('\n')[0].split(",")
        img = value[0]
        label = value[1]
        x_test.append(img)
        y_test.append(label) 
    fff.close()
    test_data = list(zip(x_test,y_test))
    return (test_data)


'''
# 生成一个训练batch
# 同时输出 reshaped train_xs and a cursor
# <Return>The third var, [start + batch_size], is a cursor that indicates current reading position, preparing for the next [start] of input</Return>
'''
def get_next_batch(T_d, start , batch_size=const.BATCH_SIZE):
    if start + batch_size > len(T_d):
        start = 0
        random.shuffle(T_d)
    logger = get_logger()
    logger.debug("start creating batch: from %d to %d with %d in batch size" % (start, start + batch_size, batch_size))
    # logger.info("the shape of the image: (%d, %d)" % (const.IMAGE_HEIGHT, const.IMAGE_WIDTH))
    batch_x = np.zeros([batch_size, const.IMAGE_HEIGHT * const.IMAGE_WIDTH])
    # batch_x = np.zeros(
    # 	[batch_size, 
    # 	const.IMAGE_HEIGHT,
    # 	const.IMAGE_WIDTH,
    # 	const.NUM_CHANNELS])

    batch_y = np.zeros([batch_size, const.MAX_CAPTCHA * const.CHAR_SET_LEN])
 
    for i in range(batch_size):
        image, text = T_d[start+i][0], T_d[start+i][1].strip()
        img = Image.open('%s/%s' % (const.ffle, image))
        arr = np.asarray(img, dtype="float32")/255.0
        
        batch_x[i,:] = np.mean(arr,-1).flatten() # (image.flatten()-128)/128  mean为0
        batch_y[i,:] = text2vec(text)
    batch_x = np.reshape(
        batch_x,
        [-1,
        const.IMAGE_HEIGHT,
        const.IMAGE_WIDTH,
        const.NUM_CHANNELS]
    )
    return batch_x, batch_y, start + batch_size
