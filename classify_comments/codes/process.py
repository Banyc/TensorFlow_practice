import os
import random 
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging

import const


def __get_lexcion():
    with open(const.LEXCION_PATH, 'rb') as f_pickle:
        lex = pickle.load(f_pickle)
    return lex


def __get_random_line(file, point):
    file.seek(point)
    file.readline()
    return file.readline()
# 从文件中随机选择n条记录
def __get_n_random_line(file_name, n=150):
    lines = []
    file = open(file_name, encoding='latin-1')
    total_bytes = os.stat(file_name).st_size 
    for i in range(n):
        random_point = random.randint(0, total_bytes)
        lines.append(__get_random_line(file, random_point))
    file.close()
    return lines
 
 
def get_test_dataset():
    lex = __get_lexcion()
    with open(const.TEST_PATH, encoding='latin-1') as f:
        test_x = []
        test_y = []
        lemmatizer = WordNetLemmatizer()
        for line in f:
            label = line.split(':%:%:%:')[0]
            tweet = line.split(':%:%:%:')[1]
            words = word_tokenize(tweet.lower())
            words = [lemmatizer.lemmatize(word) for word in words]
            features = np.zeros(len(lex))
            for word in words:
                if word in lex:
                    features[lex.index(word)] = 1
            
            test_x.append(list(features))
            test_y.append(eval(label))
    return test_x, test_y


# ONLY process and return training_data!
def get_next_batch(batch_size=150):
    logger = get_logger()
    lex = __get_lexcion()
    batch_x = []
    batch_y = []
    lemmatizer = WordNetLemmatizer()
    lines = __get_n_random_line(const.TRAIN_PATH, batch_size)
    for line in lines:
        label = line.split(':%:%:%:')[0]
        tweet = line.split(':%:%:%:')[1]
        words = word_tokenize(tweet.lower())
        words = [lemmatizer.lemmatize(word) for word in words]

        features = np.zeros(len(lex))
        for word in words:
            if word in lex:
                features[lex.index(word)] = 1  # 一个句子中某个词可能出现两次,可以用+=1，其实区别不大
    
        batch_x.append(list(features))
        batch_y.append(eval(label))
    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    # print(batch_x.shape)
    logger.debug("Batch_x_shape: " + str(batch_x.shape))
    return batch_x, batch_y


def get_lex_len():
    lex = __get_lexcion()
    return len(lex)


def get_logger():
    FORMAT = '[%(asctime)s, %(levelname)-7s]: %(message)s'

    #日志打印到文件中
    logging.basicConfig(format=FORMAT,filename=const.LOG_PATH,filemode='a')
    logger = logging.getLogger('Train')
    logger.setLevel(logging.INFO)
    # logger.setLevel(logging.DEBUG)
    return logger

def get_vec_from_text(text):
    tweet = text
    lemmatizer = WordNetLemmatizer()
    lex = __get_lexcion()
    words = word_tokenize(tweet.lower())
    words = [lemmatizer.lemmatize(word) for word in words]
    features = np.zeros(len(lex))
    for word in words:
        if word in lex:
            features[lex.index(word)] = 1
    features = np.reshape(np.array(list(features)), [-1, get_lex_len()])
    print(features.shape)
    return features
