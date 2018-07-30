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

DATA_PATH = './name/data/name.csv'

MODEL_DIR = "./name/model/"

MODEL_NAME = "name2sex.model"

NUM_CLASSES = 2

LOG_PATH = "./name/log/train.log"

EPOCH = 1
