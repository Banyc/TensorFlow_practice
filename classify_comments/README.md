# Comments Classification

original url: http://blog.topspeedsnail.com/archives/10420
Explanation: https://blog.csdn.net/u013713117/article/details/55049808

# Introduction

to classify sentiment of different comments.

Poor performance.

Only with 0.46 accuracy and about 0.59 loss in training branch with fully connected network.

While only 0.3 acc and about 0.7 and 1.8 loss respectively in training and testing branch with convolution neuron network

# Preparation

## Environment required

- package

    - TensorFlow 1.9.0

    - [Natural Language Toolkit](https://www.nltk.org/) - preprocess languages

- Anaconda

## Data set for training

- [Sentiment140](http://help.sentiment140.com/for-students/) - site of data

### What is the format of the training data?
The data is a CSV with emoticons removed. Data file format has 6 fields:
0 - the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
1 - the id of the tweet (2087)
2 - the date of the tweet (Sat May 16 23:58:44 UTC 2009)
3 - the query (lyx). If there is no query, then this value is NO_QUERY.
4 - the user that tweeted (robotickilldozr)
5 - the text of the tweet (Lyx is cool)

If you use this data, please cite Sentiment140 as your source.

# How to Run

Simply run py-files

# TODO

 - [x] finish preprocessing work

 - [ ] find out the cause of the unsatisfying output.
