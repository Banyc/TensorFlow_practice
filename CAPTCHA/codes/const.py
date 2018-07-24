# 一个常量类，用于集中常量
# origin url: https://blog.csdn.net/my_precious/article/details/50954622
#coding:utf-8

class _const:
  class ConstError(TypeError): pass
  class ConstCaseError(ConstError): pass

  def __setattr__(self, name, value):
      if name in self.__dict__:
          raise self.ConstError("can't change const %s" % name)
      if not name.isupper():
          raise self.ConstCaseError('const name "%s" is not all uppercase' % name)
      self.__dict__[name] = value

const = _const()
# const.PI = 3.14

"""
cnn在图像大小是2的倍数时性能最高, 如果你用的图像大小不是2的倍数，可以在图像边缘补无用像素。
np.pad(image,((2,3),(2,2)), 'constant', constant_values=(255,))  # 在图像上补2行，下补3行，左补2行，右补2行
"""
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
# 文本转向量
char_set = alphabet  
# 如果验证码长度小于4, '_'用来补齐，MAX_CAPTCHA=26
CHAR_SET_LEN = len(char_set)

# 图像大小
#MAX_CAPTCHA：验证码的位数
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 128
NUM_CHANNELS = 1
MAX_CAPTCHA = 4
# print("验证码文本最长字符数", MAX_CAPTCHA)   # 验证码最长4字符; 我全部固定为4,可以不固定. 如果验证码长度小于4，用'_'补齐

LOG_PATH = "./CAPTCHA/train_128.log"

# directory of training and testing data
ffle = "./CAPTCHA/CAPTCHA_data/all_data_128_64" 

# 测试库地址
TEST_PATH = "./CAPTCHA/CAPTCHA_data/test.csv"

# training set's path
TRAIN_PATH = "./CAPTCHA/CAPTCHA_data/train.csv"

MODEL_DIR = "./CAPTCHA/model"
MODEL_NAME = "crack_capcha.ckpt"

TRAINING_STEPS = 1000

REGULARIZATION_RATE = 0.01

LEARNING_RATE_BASE = 0.001
DECAY_LEARNING_RATE = 0.99

BATCH_SIZE = 64

