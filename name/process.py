#coding:utf-8


# decode('gbk').encode('utf-8')
# import tensorflow as tf 
import numpy as np 
import const
import logging


class Data_Process:
    batch_cursor = 0
    cur_epoch = 0

    def __init__(self):
        self.train_x = []
        self.train_y = []
        with open(const.DATA_PATH, 'r', encoding="utf-8") as f:
            first_line = True
            for line in f:
                if first_line is True:
                    first_line = False
                    continue
                sample = line.strip().split(',')
                if len(sample) == 2:
                    self.train_x.append(sample[0])
                    if sample[1] == '男':
                        self.train_y.append([0, 1])  # 男
                    else:
                        self.train_y.append([1, 0])  # 女
        
        self.max_name_length = max([len(name) for name in self.train_x])
        print("最长名字的字符数: ", self.max_name_length)
        self.max_name_length = 8
    
        # 数据已shuffle
        # shuffle_indices = np.random.permutation(np.arange(len(train_y)))
        # train_x = train_x[shuffle_indices]
        # train_y = train_y[shuffle_indices]
            
        
        # 词汇表（参看聊天机器人练习）
        counter = 0
        vocabulary = {}
        for name in self.train_x:
            counter += 1
            tokens = [word for word in name]
            for word in tokens:
                if word in vocabulary:
                    vocabulary[word] += 1
                else:
                    vocabulary[word] = 1
        
        vocabulary_list = [' '] + sorted(vocabulary, key=vocabulary.get, reverse=True)
        print(len(vocabulary_list))
        self.voc_len = len(vocabulary_list)

        
        # 字符串转为向量形式
        vocab = dict([(x, y) for (y, x) in enumerate(vocabulary_list)])
        self.train_x_vec = []
        for name in self.train_x:
            name_vec = []
            for word in name:
                name_vec.append(vocab.get(word))
            # while len(name_vec) < self.max_name_length:
            while len(name_vec) <= self.max_name_length:
                name_vec.append(0)
            self.train_x_vec.append(name_vec)
        self.train_x_len = len(self.train_x_vec)


    def shuffle_train(self):
        shuffle_indices = np.random.permutation(np.arange(len(self.train_y)))
        self.train_x_vec = self.train_x_vec[shuffle_indices]
        self.train_y = self.train_y[shuffle_indices]
        

    def get_train_batch(self, batch_size=64):
        num_batch = self.train_x_len // batch_size
        if self.batch_cursor + num_batch > self.train_x_len:
            self.shuffle_train()
            self.batch_cursor = 0
            self.cur_epoch += 1
        batch_xs = self.train_x_vec[self.batch_cursor:self.batch_cursor + num_batch]
        batch_ys = self.train_y[self.batch_cursor:self.batch_cursor + num_batch]
        return (batch_xs, batch_ys)


def get_logger():
    FORMAT = '[%(asctime)s, %(levelname)-7s]: %(message)s'

    #日志打印到文件中
    logging.basicConfig(format=FORMAT,filename=const.LOG_PATH,filemode='a')
    logger = logging.getLogger('Train')
    logger.setLevel(logging.INFO)
    # logger.setLevel(logging.DEBUG)
    return logger


def get_vocab(train_xs):
    # 词汇表（参看聊天机器人练习）
    counter = 0
    vocabulary = {}
    for name in train_xs:
        counter += 1
        tokens = [word for word in name]
        for word in tokens:
            if word in vocabulary:
                vocabulary[word] += 1
            else:
                vocabulary[word] = 1
    
    vocabulary_list = [' '] + sorted(vocabulary, key=vocabulary.get, reverse=True)
    return vocabulary_list

            
# for test
def get_word_vec(text_list, vocabulary_list=None, max_name_length=None):
    if vocabulary_list == None:
        Train = Data_Process()
        
        vocabulary_list = get_vocab(Train.train_x)
        max_name_length = Train.max_name_length
    # 字符串转为向量形式
    vocab = dict([(x, y) for (y, x) in enumerate(vocabulary_list)])
    x_vec = []
    for name in text_list:
        name_vec = []
        for word in name:
            name_vec.append(vocab.get(word))
        while len(name_vec) < max_name_length:
            name_vec.append(0)
        x_vec.append(name_vec)
    return (x_vec, max_name_length, len(vocabulary_list))
    