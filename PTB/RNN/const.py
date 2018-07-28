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

TRAIN_DATA = "./PTB/data/ptb.train"
TRAIN_BATCH_SIZE = 20
TRAIN_NUM_STEP = 35

EVAL_DATA = "./PTB/data/ptb.valid"
TEST_DATA = "./PTB/data/ptb.test"
HIDDEN_SIZE = 300
NUM_LAYERS = 2  # LSTM
VOCAB_SIZE = 10000

EVAL_BATCH_SIZE = 1
EVAL_NUM_STEP = 1
NUM_EPOCH = 5  # 使用train_data的轮数
LSTM_KEEP_PROB = 0.9  # LSTM节点不被dropout的概率
EMBEDDING_KEEP_PROB = 0.9
MAX_GRAD_NORM = 5  # 控制梯度膨胀的梯度大小上限
SHARE_EMB_AND_SOFTMAX = True






