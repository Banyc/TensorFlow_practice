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




# REGULARIZATION_RATE = 0.01


# DECAY_LEARNING_RATE = 0.99




'''
new
'''
TRAINING_STEPS = 500

ORG_train_file = './classify_comments/data/training.1600000.processed.noemoticon.csv'
ORG_test_file = './classify_comments/data/testdata.manual.2009.06.14.csv'



# training set's path
TRAIN_PATH = './classify_comments/data/training.csv'
# 测试库地址
TEST_PATH = './classify_comments/data/testing.csv'

# a set of lemmatized and selected words in "pickle" extension for later comparasion with training data.
LEXCION_PATH = './classify_comments/data/lexcion.pickle'

MODEL_DIR = "./classify_comments/model"
MODEL_NAME = "model.ckpt"

LEARNING_RATE_BASE = 0.001

LOG_PATH = "./classify_comments/log/train.log"

PREDICT_TEXT = "I am angry!"

TENSORBOARD_LOG_DIR = "./classify_comments/log/"
