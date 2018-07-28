import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
 
import pickle
import numpy as np
import pandas as pd
from collections import OrderedDict

# Importing files from different folder
# source: https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import sys
sys.path.insert(0, './classify_comments/codes/')

import const

# ORG_train_file = 'training.1600000.processed.noemoticon.csv'
# ORG_test_file = 'testdata.manual.2009.06.14.csv'
ORG_train_file = './classify_comments/data/training.1600000.processed.noemoticon.csv'
ORG_test_file = './classify_comments/data/testdata.manual.2009.06.14.csv'
 
# 提取文件中有用的字段
def usefull_filed(org_file, output_file):
    output = open(output_file, 'w')
    # with open(org_file, buffering=10000, encoding='latin-1') as f:
    with open(org_file, encoding='latin-1') as f:
                  
        for line in f:                # "4","2193601966","Tue Jun 16 08:40:49 PDT 2009","NO_QUERY","AmandaMarie1028","Just woke up. Having no school is the best feeling ever "
            try:
                line = line.replace('"', '')
                clf = line.split(',')[0]   # 4
                if clf == '0':
                    clf = [0, 0, 1]  # 消极评论
                elif clf == '2':
                    clf = [0, 1, 0]  # 中性评论
                elif clf == '4':
                    clf = [1, 0, 0]  # 积极评论

                tweet = line.split(',')[-1]
                outputline = str(clf) + ':%:%:%:' + tweet
                output.write(outputline)  # [0, 0, 1]:%:%:%: that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D
            except Exception as e:
                print(e)
            
    output.close()  # 处理完成，处理后文件大小127.5M
 
# usefull_filed(const.ORG_train_file, const.TRAIN_PATH)
# usefull_filed(const.ORG_test_file, const.TEST_PATH)
 
# 创建词汇表
def create_lexicon(train_file):
    lex = []
    lemmatizer = WordNetLemmatizer()
    with open(train_file, buffering=10000, encoding='latin-1') as f:
        try:
            count_word = {}  # 统计单词出现次数
            for line in f:
                tweet = line.split(':%:%:%:')[1]
                words = word_tokenize(line.lower())
                for word in words:
                    word = lemmatizer.lemmatize(word)
                    if word not in count_word:
                        count_word[word] = 1
                    else:
                        count_word[word] += 1
 
            count_word = OrderedDict(sorted(count_word.items(), key=lambda t: t[1]))
            for word in count_word:
                if count_word[word] < 100000 and count_word[word] > 100:  # 过滤掉一些词
                    lex.append(word)
        except Exception as e:
            print(e)
    return lex
 
lex = create_lexicon(const.TRAIN_PATH)
 
with open(const.LEXCION_PATH, 'wb') as f:
    pickle.dump(lex, f)
 
 
"""
# 把字符串转为向量
def string_to_vector(input_file, output_file, lex):
    output_f = open(output_file, 'w')
    lemmatizer = WordNetLemmatizer()
    with open(input_file, buffering=10000, encoding='latin-1') as f:
        for line in f:
            label = line.split(':%:%:%:')[0]
            tweet = line.split(':%:%:%:')[1]
            words = word_tokenize(tweet.lower())
            words = [lemmatizer.lemmatize(word) for word in words]
 
            features = np.zeros(len(lex))
            for word in words:
                if word in lex:
                    features[lex.index(word)] = 1  # 一个句子中某个词可能出现两次,可以用+=1，其实区别不大
            
            features = list(features)
            output_f.write(str(label) + ":" + str(features) + '\n')
    output_f.close()
 
 
f = open('lexcion.pickle', 'rb')
lex = pickle.load(f)
f.close()
 
# lexcion词汇表大小112k,training.vec大约112k*1600000  170G  太大，只能边转边训练了, so the function is aborted.
# string_to_vector('training.csv', 'training.vec', lex)
# string_to_vector('tesing.csv', 'tesing.vec', lex)
"""
