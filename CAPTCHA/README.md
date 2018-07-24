# Identify CAPTCHAs

References (original urls): https://blog.csdn.net/u011684265/article/details/77776151, http://blog.topspeedsnail.com/archives/10858

# Introduction

The network preforms ill with a lack of training data set. 

Train is still under processing. 

Final accuracy is about 0.2.

Suggestions are greatly needed.

## Network config

- convolution_1:
    - size: 3 * 3
    - deep: 32
- pooling_1:
    - size: 2 * 2
    - step: 2

- convolution_2:
    - size: 3 * 3
    - deep: 64
- pooling_2:
    - size: 2 * 2
    - step: 2

- convolution_3:
    - size: 3 * 3
    - deep: 64
- pooling_3:
    - size: 2 * 2
    - step: 2

- full_connection_1:
    - size: 1024

- full_connection_2:
    - output

## Training config

- loss consists of sigmoid_cross_entropy_with_logits.

- includes a learning rate self-decay.

- optimizer is AdamOptimizer.

- no l2_regularizer included.

- with only 5000 training images from unknown source

## Files info

- "./codes/train.py" is for training data.

- "./train_128.log" is a log.

- "./codes/evaluate.py"

- "./codes/process.py" contains shared functions of remaining py files.

- "./codes/const.py" is a set of shared constants.

- "./all_data_128_64/" contains testing and training data.

- "./model/" saved model.
