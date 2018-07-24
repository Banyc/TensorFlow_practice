import tensorflow as tf 
import numpy as np 
import os

import inference

from tensorflow.examples.tutorials.mnist import input_data

MODEL_PATH = "./test/LeNet-5_test/model"
MODEL_NAME = "model.ckpt"

REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 2000
BATCH_SIZE = 500

LEARNING_RATE_BASE = 0.001
DECAY_RATE = 0.99


def train(mnist):
    x = tf.placeholder(
        tf.float32,
        [None,
        inference.IMAGE_SIZE,
        inference.IMAGE_SIZE,
        inference.NUM_CHANNELS],
        "x-input"
    )
    y_ = tf.placeholder(
        tf.float32,
        [None, inference.NUM_LABELS],
        "y_-input"
    )
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    Is_train = True

    y = inference.inference(x, Is_train, regularizer)

    global_step = tf.Variable(
        0, False
    )

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.argmax(y_, 1), logits=y
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, 
        global_step, 
        mnist.train.num_examples / BATCH_SIZE, 
        DECAY_RATE
    )

    # train_step = tf.train.AdamOptimizer().minimize(loss)
    train_step = tf.train.GradientDescentOptimizer(
        learning_rate
    ).minimize(
        loss, global_step=global_step
    )

    with tf.control_dependencies([train_step]):
        train_op = tf.no_op("train")

    # prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        
        # validate_xs = mnist.validation.images[:50]
        # validate_xs_reshaped = np.reshape(  # tf.reshape() is unacceptable
        #     validate_xs,
        #     [-1,
        #     inference.IMAGE_SIZE,
        #     inference.IMAGE_SIZE,
        #     inference.NUM_CHANNELS])
        for i in range(TRAINING_STEPS):
            train_xs, train_ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_train_xs = np.reshape(  # tf.reshape() is unacceptable
                train_xs,
                [-1, 
                inference.IMAGE_SIZE,
                inference.IMAGE_SIZE,
                inference.NUM_CHANNELS]
            )
            print("train step: ", i)
            print("global_step: ", sess.run(global_step))
            print("learning rate: ", sess.run(learning_rate))
            print("==========")
            # Is_train = True
            sess.run(train_op, {x: reshaped_train_xs, y_: train_ys})
            if i % 50 == 0:
            #     Is_train = False
                # TODO unknown BUG; solution: multi-files
            #     # issue solved: https://blog.csdn.net/jt31520/article/details/71411335
            #     validate_acc = sess.run(accuracy, {x: validate_xs_reshaped, y_: mnist.validation.labels[:50]})
            #     print(i, validate_acc)
                saver.save(
                    sess, os.path.join(MODEL_PATH, MODEL_NAME)
                )


def main(argv=None):
    mnist = input_data.read_data_sets("./mnist", one_hot=True)
    train(mnist)
    return 0


if __name__ == "__main__":
    # tf.app.run()
    main()

# OUTPUT
# [Running] python "c:\Users\fsXian\PycharmProjects\TensorFlow\TensorFlow实战Google深度学习框架\mnist识别\test\LeNet-5_test\eval.py"
# 2018-07-15 12:05:07.180055: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE instructions, but these are available on your machine and could speed up CPU computations.
# 2018-07-15 12:05:07.183894: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE2 instructions, but these are available on your machine and could speed up CPU computations.
# 2018-07-15 12:05:07.187403: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE3 instructions, but these are available on your machine and could speed up CPU computations.
# 2018-07-15 12:05:07.190964: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
# 2018-07-15 12:05:07.194191: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
# 2018-07-15 12:05:07.197651: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
# 2018-07-15 12:05:07.202639: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
# 2018-07-15 12:05:07.206088: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
# D:\ProgramData\Anaconda3\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
#   from ._conv import register_converters as _register_converters
# Extracting ./mnist\train-images-idx3-ubyte.gz
# Extracting ./mnist\train-labels-idx1-ubyte.gz
# Extracting ./mnist\t10k-images-idx3-ubyte.gz
# Extracting ./mnist\t10k-labels-idx1-ubyte.gz
# 0.9738

# [Done] exited with code=0 in 128.309 seconds

# [Running] python "c:\Users\fsXian\PycharmProjects\TensorFlow\TensorFlow实战Google深度学习框架\mnist识别\test\LeNet-5_test\train.py"
# 2018-07-15 13:45:19.247721: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE instructions, but these are available on your machine and could speed up CPU computations.
# 2018-07-15 13:45:19.251929: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE2 instructions, but these are available on your machine and could speed up CPU computations.
# 2018-07-15 13:45:19.255678: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE3 instructions, but these are available on your machine and could speed up CPU computations.
# 2018-07-15 13:45:19.259351: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
# 2018-07-15 13:45:19.262955: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
# 2018-07-15 13:45:19.266222: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
# 2018-07-15 13:45:19.269678: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
# 2018-07-15 13:45:19.273404: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
# Extracting ./mnist\train-images-idx3-ubyte.gz
# Extracting ./mnist\train-labels-idx1-ubyte.gz
# Extracting ./mnist\t10k-images-idx3-ubyte.gz
# Extracting ./mnist\t10k-labels-idx1-ubyte.gz
# train step:  0
# global_step:  0
# learning rate:  0.001
# train step:  1
# global_step:  1
# learning rate:  0.0009999087
# train step:  2
# global_step:  2
# learning rate:  0.0009998174
# train step:  3
# global_step:  3
# learning rate:  0.000999726
# train step:  4
# global_step:  4
# learning rate:  0.0009996346
# train step:  5
# global_step:  5
# learning rate:  0.0009995434
# train step:  6
# global_step:  6
# learning rate:  0.000999452
# train step:  7
# global_step:  7
# learning rate:  0.0009993607
# train step:  8
# global_step:  8
# learning rate:  0.0009992693
# train step:  9
# global_step:  9
# learning rate:  0.0009991782
# train step:  10
# global_step:  10
# learning rate:  0.0009990868
# train step:  11
# global_step:  11
# learning rate:  0.0009989955
# train step:  12
# global_step:  12
# learning rate:  0.0009989042
# train step:  13
# global_step:  13
# learning rate:  0.000998813
# train step:  14
# global_step:  14
# learning rate:  0.0009987217
# train step:  15
# global_step:  15
# learning rate:  0.0009986305
# train step:  16
# global_step:  16
# learning rate:  0.0009985393
# train step:  17
# global_step:  17
# learning rate:  0.000998448
# train step:  18
# global_step:  18
# learning rate:  0.0009983568
# train step:  19
# global_step:  19
# learning rate:  0.0009982656
# train step:  20
# global_step:  20
# learning rate:  0.0009981743
# train step:  21
# global_step:  21
# learning rate:  0.0009980832
# train step:  22
# global_step:  22
# learning rate:  0.000997992
# train step:  23
# global_step:  23
# learning rate:  0.0009979008
# train step:  24
# global_step:  24
# learning rate:  0.0009978097
# train step:  25
# global_step:  25
# learning rate:  0.0009977185
# train step:  26
# global_step:  26
# learning rate:  0.0009976273
# train step:  27
# global_step:  27
# learning rate:  0.0009975361
# train step:  28
# global_step:  28
# learning rate:  0.0009974451
# train step:  29
# global_step:  29
# learning rate:  0.0009973539
# train step:  30
# global_step:  30
# learning rate:  0.0009972628
# train step:  31
# global_step:  31
# learning rate:  0.0009971717
# train step:  32
# global_step:  32
# learning rate:  0.0009970806
# train step:  33
# global_step:  33
# learning rate:  0.0009969894
# train step:  34
# global_step:  34
# learning rate:  0.0009968984
# train step:  35
# global_step:  35
# learning rate:  0.0009968074
# train step:  36
# global_step:  36
# learning rate:  0.0009967162
# train step:  37
# global_step:  37
# learning rate:  0.0009966252
# train step:  38
# global_step:  38
# learning rate:  0.0009965341
# train step:  39
# global_step:  39
# learning rate:  0.0009964431
# train step:  40
# global_step:  40
# learning rate:  0.000996352
# train step:  41
# global_step:  41
# learning rate:  0.000996261
# train step:  42
# global_step:  42
# learning rate:  0.00099617
# train step:  43
# global_step:  43
# learning rate:  0.0009960791
# train step:  44
# global_step:  44
# learning rate:  0.000995988
# train step:  45
# global_step:  45
# learning rate:  0.000995897
# train step:  46
# global_step:  46
# learning rate:  0.0009958061
# train step:  47
# global_step:  47
# learning rate:  0.000995715
# train step:  48
# global_step:  48
# learning rate:  0.000995624
# train step:  49
# global_step:  49
# learning rate:  0.0009955331
# train step:  50
# global_step:  50
# learning rate:  0.0009954422
# train step:  51
# global_step:  51
# learning rate:  0.0009953512
# train step:  52
# global_step:  52
# learning rate:  0.0009952602
# train step:  53
# global_step:  53
# learning rate:  0.0009951693
# train step:  54
# global_step:  54
# learning rate:  0.0009950784
# train step:  55
# global_step:  55
# learning rate:  0.0009949874
# train step:  56
# global_step:  56
# learning rate:  0.0009948966
# train step:  57
# global_step:  57
# learning rate:  0.0009948057
# train step:  58
# global_step:  58
# learning rate:  0.0009947148
# train step:  59
# global_step:  59
# learning rate:  0.000994624
# train step:  60
# global_step:  60
# learning rate:  0.0009945331
# train step:  61
# global_step:  61
# learning rate:  0.0009944423
# train step:  62
# global_step:  62
# learning rate:  0.0009943513
# train step:  63
# global_step:  63
# learning rate:  0.0009942604
# train step:  64
# global_step:  64
# learning rate:  0.0009941696
# train step:  65
# global_step:  65
# learning rate:  0.0009940788
# train step:  66
# global_step:  66
# learning rate:  0.000993988
# train step:  67
# global_step:  67
# learning rate:  0.0009938972
# train step:  68
# global_step:  68
# learning rate:  0.0009938064
# train step:  69
# global_step:  69
# learning rate:  0.0009937156
# train step:  70
# global_step:  70
# learning rate:  0.0009936248
# train step:  71
# global_step:  71
# learning rate:  0.000993534
# train step:  72
# global_step:  72
# learning rate:  0.0009934432
# train step:  73
# global_step:  73
# learning rate:  0.0009933525
# train step:  74
# global_step:  74
# learning rate:  0.0009932617
# train step:  75
# global_step:  75
# learning rate:  0.000993171
# train step:  76
# global_step:  76
# learning rate:  0.0009930802
# train step:  77
# global_step:  77
# learning rate:  0.0009929895
# train step:  78
# global_step:  78
# learning rate:  0.0009928988
# train step:  79
# global_step:  79
# learning rate:  0.0009928081
# train step:  80
# global_step:  80
# learning rate:  0.0009927173
# train step:  81
# global_step:  81
# learning rate:  0.0009926267
# train step:  82
# global_step:  82
# learning rate:  0.000992536
# train step:  83
# global_step:  83
# learning rate:  0.0009924453
# train step:  84
# global_step:  84
# learning rate:  0.0009923546
# train step:  85
# global_step:  85
# learning rate:  0.000992264
# train step:  86
# global_step:  86
# learning rate:  0.0009921733
# train step:  87
# global_step:  87
# learning rate:  0.0009920826
# train step:  88
# global_step:  88
# learning rate:  0.0009919921
# train step:  89
# global_step:  89
# learning rate:  0.0009919014
# train step:  90
# global_step:  90
# learning rate:  0.0009918108
# train step:  91
# global_step:  91
# learning rate:  0.0009917201
# train step:  92
# global_step:  92
# learning rate:  0.0009916296
# train step:  93
# global_step:  93
# learning rate:  0.000991539
# train step:  94
# global_step:  94
# learning rate:  0.0009914484
# train step:  95
# global_step:  95
# learning rate:  0.0009913578
# train step:  96
# global_step:  96
# learning rate:  0.0009912672
# train step:  97
# global_step:  97
# learning rate:  0.0009911767
# train step:  98
# global_step:  98
# learning rate:  0.0009910861
# train step:  99
# global_step:  99
# learning rate:  0.0009909956
# train step:  100
# global_step:  100
# learning rate:  0.000990905
# train step:  101
# global_step:  101
# learning rate:  0.0009908145
# train step:  102
# global_step:  102
# learning rate:  0.000990724
# train step:  103
# global_step:  103
# learning rate:  0.0009906334
# train step:  104
# global_step:  104
# learning rate:  0.0009905429
# train step:  105
# global_step:  105
# learning rate:  0.0009904524
# train step:  106
# global_step:  106
# learning rate:  0.0009903619
# train step:  107
# global_step:  107
# learning rate:  0.0009902715
# train step:  108
# global_step:  108
# learning rate:  0.000990181
# train step:  109
# global_step:  109
# learning rate:  0.0009900905
# train step:  110
# global_step:  110
# learning rate:  0.0009900001
# train step:  111
# global_step:  111
# learning rate:  0.0009899096
# train step:  112
# global_step:  112
# learning rate:  0.0009898192
# train step:  113
# global_step:  113
# learning rate:  0.0009897287
# train step:  114
# global_step:  114
# learning rate:  0.0009896383
# train step:  115
# global_step:  115
# learning rate:  0.0009895479
# train step:  116
# global_step:  116
# learning rate:  0.0009894575
# train step:  117
# global_step:  117
# learning rate:  0.0009893671
# train step:  118
# global_step:  118
# learning rate:  0.0009892767
# train step:  119
# global_step:  119
# learning rate:  0.0009891863
# train step:  120
# global_step:  120
# learning rate:  0.0009890959
# train step:  121
# global_step:  121
# learning rate:  0.0009890056
# train step:  122
# global_step:  122
# learning rate:  0.0009889152
# train step:  123
# global_step:  123
# learning rate:  0.0009888249
# train step:  124
# global_step:  124
# learning rate:  0.0009887345
# train step:  125
# global_step:  125
# learning rate:  0.0009886442
# train step:  126
# global_step:  126
# learning rate:  0.0009885539
# train step:  127
# global_step:  127
# learning rate:  0.0009884635
# train step:  128
# global_step:  128
# learning rate:  0.0009883733
# train step:  129
# global_step:  129
# learning rate:  0.000988283
# train step:  130
# global_step:  130
# learning rate:  0.0009881926
# train step:  131
# global_step:  131
# learning rate:  0.0009881024
# train step:  132
# global_step:  132
# learning rate:  0.0009880121
# train step:  133
# global_step:  133
# learning rate:  0.0009879218
# train step:  134
# global_step:  134
# learning rate:  0.0009878316
# train step:  135
# global_step:  135
# learning rate:  0.0009877413
# train step:  136
# global_step:  136
# learning rate:  0.0009876511
# train step:  137
# global_step:  137
# learning rate:  0.0009875608
# train step:  138
# global_step:  138
# learning rate:  0.0009874706
# train step:  139
# global_step:  139
# learning rate:  0.0009873804
# train step:  140
# global_step:  140
# learning rate:  0.0009872902
# train step:  141
# global_step:  141
# learning rate:  0.0009872001
# train step:  142
# global_step:  142
# learning rate:  0.0009871097
# train step:  143
# global_step:  143
# learning rate:  0.0009870196
# train step:  144
# global_step:  144
# learning rate:  0.0009869294
# train step:  145
# global_step:  145
# learning rate:  0.0009868393
# train step:  146
# global_step:  146
# learning rate:  0.0009867491
# train step:  147
# global_step:  147
# learning rate:  0.000986659
# train step:  148
# global_step:  148
# learning rate:  0.0009865687
# train step:  149
# global_step:  149
# learning rate:  0.0009864786
# train step:  150
# global_step:  150
# learning rate:  0.0009863885
# train step:  151
# global_step:  151
# learning rate:  0.0009862984
# train step:  152
# global_step:  152
# learning rate:  0.0009862083
# train step:  153
# global_step:  153
# learning rate:  0.0009861182
# train step:  154
# global_step:  154
# learning rate:  0.0009860281
# train step:  155
# global_step:  155
# learning rate:  0.000985938
# train step:  156
# global_step:  156
# learning rate:  0.0009858479
# train step:  157
# global_step:  157
# learning rate:  0.0009857579
# train step:  158
# global_step:  158
# learning rate:  0.0009856678
# train step:  159
# global_step:  159
# learning rate:  0.0009855777
# train step:  160
# global_step:  160
# learning rate:  0.0009854877
# train step:  161
# global_step:  161
# learning rate:  0.0009853977
# train step:  162
# global_step:  162
# learning rate:  0.0009853077
# train step:  163
# global_step:  163
# learning rate:  0.0009852176
# train step:  164
# global_step:  164
# learning rate:  0.0009851276
# train step:  165
# global_step:  165
# learning rate:  0.0009850377
# train step:  166
# global_step:  166
# learning rate:  0.0009849477
# train step:  167
# global_step:  167
# learning rate:  0.0009848577
# train step:  168
# global_step:  168
# learning rate:  0.0009847677
# train step:  169
# global_step:  169
# learning rate:  0.0009846777
# train step:  170
# global_step:  170
# learning rate:  0.0009845877
# train step:  171
# global_step:  171
# learning rate:  0.0009844977
# train step:  172
# global_step:  172
# learning rate:  0.0009844078
# train step:  173
# global_step:  173
# learning rate:  0.0009843179
# train step:  174
# global_step:  174
# learning rate:  0.000984228
# train step:  175
# global_step:  175
# learning rate:  0.0009841381
# train step:  176
# global_step:  176
# learning rate:  0.0009840481
# train step:  177
# global_step:  177
# learning rate:  0.0009839582
# train step:  178
# global_step:  178
# learning rate:  0.0009838684
# train step:  179
# global_step:  179
# learning rate:  0.0009837785
# train step:  180
# global_step:  180
# learning rate:  0.0009836886
# train step:  181
# global_step:  181
# learning rate:  0.0009835986
# train step:  182
# global_step:  182
# learning rate:  0.0009835088
# train step:  183
# global_step:  183
# learning rate:  0.000983419
# train step:  184
# global_step:  184
# learning rate:  0.0009833291
# train step:  185
# global_step:  185
# learning rate:  0.0009832393
# train step:  186
# global_step:  186
# learning rate:  0.0009831495
# train step:  187
# global_step:  187
# learning rate:  0.0009830596
# train step:  188
# global_step:  188
# learning rate:  0.0009829698
# train step:  189
# global_step:  189
# learning rate:  0.00098288
# train step:  190
# global_step:  190
# learning rate:  0.0009827903
# train step:  191
# global_step:  191
# learning rate:  0.0009827004
# train step:  192
# global_step:  192
# learning rate:  0.0009826106
# train step:  193
# global_step:  193
# learning rate:  0.0009825209
# train step:  194
# global_step:  194
# learning rate:  0.0009824311
# train step:  195
# global_step:  195
# learning rate:  0.0009823414
# train step:  196
# global_step:  196
# learning rate:  0.0009822516
# train step:  197
# global_step:  197
# learning rate:  0.0009821618
# train step:  198
# global_step:  198
# learning rate:  0.0009820721
# train step:  199
# global_step:  199
# learning rate:  0.0009819824
# train step:  200
# global_step:  200
# learning rate:  0.0009818927
# train step:  201
# global_step:  201
# learning rate:  0.0009818029
# train step:  202
# global_step:  202
# learning rate:  0.0009817133
# train step:  203
# global_step:  203
# learning rate:  0.0009816235
# train step:  204
# global_step:  204
# learning rate:  0.0009815339
# train step:  205
# global_step:  205
# learning rate:  0.0009814443
# train step:  206
# global_step:  206
# learning rate:  0.0009813545
# train step:  207
# global_step:  207
# learning rate:  0.0009812649
# train step:  208
# global_step:  208
# learning rate:  0.0009811752
# train step:  209
# global_step:  209
# learning rate:  0.0009810856
# train step:  210
# global_step:  210
# learning rate:  0.000980996
# train step:  211
# global_step:  211
# learning rate:  0.0009809063
# train step:  212
# global_step:  212
# learning rate:  0.0009808167
# train step:  213
# global_step:  213
# learning rate:  0.0009807271
# train step:  214
# global_step:  214
# learning rate:  0.0009806375
# train step:  215
# global_step:  215
# learning rate:  0.0009805479
# train step:  216
# global_step:  216
# learning rate:  0.0009804583
# train step:  217
# global_step:  217
# learning rate:  0.0009803687
# train step:  218
# global_step:  218
# learning rate:  0.0009802792
# train step:  219
# global_step:  219
# learning rate:  0.0009801897
# train step:  220
# global_step:  220
# learning rate:  0.0009801001
# train step:  221
# global_step:  221
# learning rate:  0.0009800105
# train step:  222
# global_step:  222
# learning rate:  0.000979921
# train step:  223
# global_step:  223
# learning rate:  0.0009798314
# train step:  224
# global_step:  224
# learning rate:  0.0009797419
# train step:  225
# global_step:  225
# learning rate:  0.0009796524
# train step:  226
# global_step:  226
# learning rate:  0.0009795629
# train step:  227
# global_step:  227
# learning rate:  0.0009794733
# train step:  228
# global_step:  228
# learning rate:  0.0009793839
# train step:  229
# global_step:  229
# learning rate:  0.0009792944
# train step:  230
# global_step:  230
# learning rate:  0.000979205
# train step:  231
# global_step:  231
# learning rate:  0.0009791155
# train step:  232
# global_step:  232
# learning rate:  0.0009790261
# train step:  233
# global_step:  233
# learning rate:  0.0009789366
# train step:  234
# global_step:  234
# learning rate:  0.0009788472
# train step:  235
# global_step:  235
# learning rate:  0.0009787577
# train step:  236
# global_step:  236
# learning rate:  0.0009786683
# train step:  237
# global_step:  237
# learning rate:  0.0009785789
# train step:  238
# global_step:  238
# learning rate:  0.0009784895
# train step:  239
# global_step:  239
# learning rate:  0.0009784001
# train step:  240
# global_step:  240
# learning rate:  0.0009783107
# train step:  241
# global_step:  241
# learning rate:  0.0009782213
# train step:  242
# global_step:  242
# learning rate:  0.000978132
# train step:  243
# global_step:  243
# learning rate:  0.0009780426
# train step:  244
# global_step:  244
# learning rate:  0.0009779532
# train step:  245
# global_step:  245
# learning rate:  0.0009778639
# train step:  246
# global_step:  246
# learning rate:  0.0009777745
# train step:  247
# global_step:  247
# learning rate:  0.0009776852
# train step:  248
# global_step:  248
# learning rate:  0.0009775959
# train step:  249
# global_step:  249
# learning rate:  0.0009775066
# train step:  250
# global_step:  250
# learning rate:  0.0009774172
# train step:  251
# global_step:  251
# learning rate:  0.000977328
# train step:  252
# global_step:  252
# learning rate:  0.0009772388
# train step:  253
# global_step:  253
# learning rate:  0.0009771495
# train step:  254
# global_step:  254
# learning rate:  0.0009770602
# train step:  255
# global_step:  255
# learning rate:  0.0009769709
# train step:  256
# global_step:  256
# learning rate:  0.0009768816
# train step:  257
# global_step:  257
# learning rate:  0.0009767924
# train step:  258
# global_step:  258
# learning rate:  0.0009767031
# train step:  259
# global_step:  259
# learning rate:  0.0009766138
# train step:  260
# global_step:  260
# learning rate:  0.00097652467
# train step:  261
# global_step:  261
# learning rate:  0.0009764355
# train step:  262
# global_step:  262
# learning rate:  0.00097634626
# train step:  263
# global_step:  263
# learning rate:  0.000976257
# train step:  264
# global_step:  264
# learning rate:  0.00097616785
# train step:  265
# global_step:  265
# learning rate:  0.0009760787
# train step:  266
# global_step:  266
# learning rate:  0.0009759895
# train step:  267
# global_step:  267
# learning rate:  0.00097590033
# train step:  268
# global_step:  268
# learning rate:  0.00097581116
# train step:  269
# global_step:  269
# learning rate:  0.000975722
# train step:  270
# global_step:  270
# learning rate:  0.0009756328
# train step:  271
# global_step:  271
# learning rate:  0.00097554375
# train step:  272
# global_step:  272
# learning rate:  0.00097545463
# train step:  273
# global_step:  273
# learning rate:  0.00097536546
# train step:  274
# global_step:  274
# learning rate:  0.00097527634
# train step:  275
# global_step:  275
# learning rate:  0.0009751872
# train step:  276
# global_step:  276
# learning rate:  0.00097509817
# train step:  277
# global_step:  277
# learning rate:  0.00097500905
# train step:  278
# global_step:  278
# learning rate:  0.00097492
# train step:  279
# global_step:  279
# learning rate:  0.00097483094
# train step:  280
# global_step:  280
# learning rate:  0.0009747419
# train step:  281
# global_step:  281
# learning rate:  0.0009746528
# train step:  282
# global_step:  282
# learning rate:  0.00097456377
# train step:  283
# global_step:  283
# learning rate:  0.0009744747
# train step:  284
# global_step:  284
# learning rate:  0.00097438565
# train step:  285
# global_step:  285
# learning rate:  0.00097429665
# train step:  286
# global_step:  286
# learning rate:  0.00097420765
# train step:  287
# global_step:  287
# learning rate:  0.00097411865
# train step:  288
# global_step:  288
# learning rate:  0.00097402965
# train step:  289
# global_step:  289
# learning rate:  0.00097394065
# train step:  290
# global_step:  290
# learning rate:  0.00097385165
# train step:  291
# global_step:  291
# learning rate:  0.00097376265
# train step:  292
# global_step:  292
# learning rate:  0.00097367377
# train step:  293
# global_step:  293
# learning rate:  0.00097358477
# train step:  294
# global_step:  294
# learning rate:  0.00097349583
# train step:  295
# global_step:  295
# learning rate:  0.0009734069
# train step:  296
# global_step:  296
# learning rate:  0.00097331795
# train step:  297
# global_step:  297
# learning rate:  0.00097322906
# train step:  298
# global_step:  298
# learning rate:  0.0009731401
# train step:  299
# global_step:  299
# learning rate:  0.00097305124
# train step:  300
# global_step:  300
# learning rate:  0.0009729623
# train step:  301
# global_step:  301
# learning rate:  0.0009728734
# train step:  302
# global_step:  302
# learning rate:  0.00097278453
# train step:  303
# global_step:  303
# learning rate:  0.00097269565
# train step:  304
# global_step:  304
# learning rate:  0.00097260677
# train step:  305
# global_step:  305
# learning rate:  0.00097251794
# train step:  306
# global_step:  306
# learning rate:  0.00097242906
# train step:  307
# global_step:  307
# learning rate:  0.00097234023
# train step:  308
# global_step:  308
# learning rate:  0.0009722514
# train step:  309
# global_step:  309
# learning rate:  0.0009721626
# train step:  310
# global_step:  310
# learning rate:  0.00097207376
# train step:  311
# global_step:  311
# learning rate:  0.000971985
# train step:  312
# global_step:  312
# learning rate:  0.00097189617
# train step:  313
# global_step:  313
# learning rate:  0.00097180734
# train step:  314
# global_step:  314
# learning rate:  0.0009717185
# train step:  315
# global_step:  315
# learning rate:  0.0009716298
# train step:  316
# global_step:  316
# learning rate:  0.00097154104
# train step:  317
# global_step:  317
# learning rate:  0.0009714522
# train step:  318
# global_step:  318
# learning rate:  0.00097136345
# train step:  319
# global_step:  319
# learning rate:  0.00097127474
# train step:  320
# global_step:  320
# learning rate:  0.00097118603
# train step:  321
# global_step:  321
# learning rate:  0.00097109727
# train step:  322
# global_step:  322
# learning rate:  0.00097100856
# train step:  323
# global_step:  323
# learning rate:  0.00097091985
# train step:  324
# global_step:  324
# learning rate:  0.00097083114
# train step:  325
# global_step:  325
# learning rate:  0.00097074243
# train step:  326
# global_step:  326
# learning rate:  0.0009706538
# train step:  327
# global_step:  327
# learning rate:  0.0009705651
# train step:  328
# global_step:  328
# learning rate:  0.00097047637
# train step:  329
# global_step:  329
# learning rate:  0.0009703877
# train step:  330
# global_step:  330
# learning rate:  0.00097029906
# train step:  331
# global_step:  331
# learning rate:  0.0009702104
# train step:  332
# global_step:  332
# learning rate:  0.00097012176
# train step:  333
# global_step:  333
# learning rate:  0.0009700332
# train step:  334
# global_step:  334
# learning rate:  0.0009699445
# train step:  335
# global_step:  335
# learning rate:  0.0009698559
# train step:  336
# global_step:  336
# learning rate:  0.00096976734
# train step:  337
# global_step:  337
# learning rate:  0.0009696787
# train step:  338
# global_step:  338
# learning rate:  0.0009695901
# train step:  339
# global_step:  339
# learning rate:  0.00096950156
# train step:  340
# global_step:  340
# learning rate:  0.00096941297
# train step:  341
# global_step:  341
# learning rate:  0.0009693244
# train step:  342
# global_step:  342
# learning rate:  0.00096923584
# train step:  343
# global_step:  343
# learning rate:  0.00096914725
# train step:  344
# global_step:  344
# learning rate:  0.0009690587
# train step:  345
# global_step:  345
# learning rate:  0.0009689702
# train step:  346
# global_step:  346
# learning rate:  0.00096888165
# train step:  347
# global_step:  347
# learning rate:  0.0009687931
# train step:  348
# global_step:  348
# learning rate:  0.00096870464
# train step:  349
# global_step:  349
# learning rate:  0.0009686161
# train step:  350
# global_step:  350
# learning rate:  0.0009685277
# train step:  351
# global_step:  351
# learning rate:  0.00096843916
# train step:  352
# global_step:  352
# learning rate:  0.0009683507
# train step:  353
# global_step:  353
# learning rate:  0.0009682622
# train step:  354
# global_step:  354
# learning rate:  0.0009681737
# train step:  355
# global_step:  355
# learning rate:  0.00096808525
# train step:  356
# global_step:  356
# learning rate:  0.00096799684
# train step:  357
# global_step:  357
# learning rate:  0.00096790836
# train step:  358
# global_step:  358
# learning rate:  0.00096782
# train step:  359
# global_step:  359
# learning rate:  0.0009677315
# train step:  360
# global_step:  360
# learning rate:  0.0009676431
# train step:  361
# global_step:  361
# learning rate:  0.00096755475
# train step:  362
# global_step:  362
# learning rate:  0.00096746633
# train step:  363
# global_step:  363
# learning rate:  0.000967378
# train step:  364
# global_step:  364
# learning rate:  0.00096728955
# train step:  365
# global_step:  365
# learning rate:  0.00096720114
# train step:  366
# global_step:  366
# learning rate:  0.00096711284
# train step:  367
# global_step:  367
# learning rate:  0.0009670244
# train step:  368
# global_step:  368
# learning rate:  0.0009669361
# train step:  369
# global_step:  369
# learning rate:  0.00096684776
# train step:  370
# global_step:  370
# learning rate:  0.00096675946
# train step:  371
# global_step:  371
# learning rate:  0.0009666711
# train step:  372
# global_step:  372
# learning rate:  0.00096658274
# train step:  373
# global_step:  373
# learning rate:  0.0009664945
# train step:  374
# global_step:  374
# learning rate:  0.00096640614
# train step:  375
# global_step:  375
# learning rate:  0.0009663179
# train step:  376
# global_step:  376
# learning rate:  0.0009662296
# train step:  377
# global_step:  377
# learning rate:  0.00096614135
# train step:  378
# global_step:  378
# learning rate:  0.00096605305
# train step:  379
# global_step:  379
# learning rate:  0.0009659648
# train step:  380
# global_step:  380
# learning rate:  0.00096587656
# train step:  381
# global_step:  381
# learning rate:  0.0009657883
# train step:  382
# global_step:  382
# learning rate:  0.0009657001
# train step:  383
# global_step:  383
# learning rate:  0.0009656118
# train step:  384
# global_step:  384
# learning rate:  0.0009655236
# train step:  385
# global_step:  385
# learning rate:  0.00096543535
# train step:  386
# global_step:  386
# learning rate:  0.0009653472
# train step:  387
# global_step:  387
# learning rate:  0.000965259
# train step:  388
# global_step:  388
# learning rate:  0.0009651708
# train step:  389
# global_step:  389
# learning rate:  0.0009650826
# train step:  390
# global_step:  390
# learning rate:  0.0009649945
# train step:  391
# global_step:  391
# learning rate:  0.00096490624
# train step:  392
# global_step:  392
# learning rate:  0.0009648181
# train step:  393
# global_step:  393
# learning rate:  0.00096473
# train step:  394
# global_step:  394
# learning rate:  0.00096464186
# train step:  395
# global_step:  395
# learning rate:  0.0009645537
# train step:  396
# global_step:  396
# learning rate:  0.0009644656
# train step:  397
# global_step:  397
# learning rate:  0.0009643774
# train step:  398
# global_step:  398
# learning rate:  0.00096428936
# train step:  399
# global_step:  399
# learning rate:  0.00096420123
# train step:  400
# global_step:  400
# learning rate:  0.00096411316
# train step:  401
# global_step:  401
# learning rate:  0.0009640251
# train step:  402
# global_step:  402
# learning rate:  0.000963937
# train step:  403
# global_step:  403
# learning rate:  0.00096384896
# train step:  404
# global_step:  404
# learning rate:  0.0009637609
# train step:  405
# global_step:  405
# learning rate:  0.0009636729
# train step:  406
# global_step:  406
# learning rate:  0.00096358475
# train step:  407
# global_step:  407
# learning rate:  0.00096349674
# train step:  408
# global_step:  408
# learning rate:  0.00096340873
# train step:  409
# global_step:  409
# learning rate:  0.0009633207
# train step:  410
# global_step:  410
# learning rate:  0.00096323265
# train step:  411
# global_step:  411
# learning rate:  0.0009631447
# train step:  412
# global_step:  412
# learning rate:  0.0009630567
# train step:  413
# global_step:  413
# learning rate:  0.0009629687
# train step:  414
# global_step:  414
# learning rate:  0.00096288073
# train step:  415
# global_step:  415
# learning rate:  0.0009627927
# train step:  416
# global_step:  416
# learning rate:  0.0009627048
# train step:  417
# global_step:  417
# learning rate:  0.0009626169
# train step:  418
# global_step:  418
# learning rate:  0.00096252887
# train step:  419
# global_step:  419
# learning rate:  0.000962441
# train step:  420
# global_step:  420
# learning rate:  0.000962353
# train step:  421
# global_step:  421
# learning rate:  0.0009622651
# train step:  422
# global_step:  422
# learning rate:  0.0009621772
# train step:  423
# global_step:  423
# learning rate:  0.0009620893
# train step:  424
# global_step:  424
# learning rate:  0.0009620014
# train step:  425
# global_step:  425
# learning rate:  0.0009619135
# train step:  426
# global_step:  426
# learning rate:  0.0009618256
# train step:  427
# global_step:  427
# learning rate:  0.00096173777
# train step:  428
# global_step:  428
# learning rate:  0.0009616499
# train step:  429
# global_step:  429
# learning rate:  0.00096156204
# train step:  430
# global_step:  430
# learning rate:  0.00096147414
# train step:  431
# global_step:  431
# learning rate:  0.0009613863
# train step:  432
# global_step:  432
# learning rate:  0.0009612985
# train step:  433
# global_step:  433
# learning rate:  0.00096121064
# train step:  434
# global_step:  434
# learning rate:  0.00096112286
# train step:  435
# global_step:  435
# learning rate:  0.000961035
# train step:  436
# global_step:  436
# learning rate:  0.0009609472
# train step:  437
# global_step:  437
# learning rate:  0.0009608594
# train step:  438
# global_step:  438
# learning rate:  0.0009607717
# train step:  439
# global_step:  439
# learning rate:  0.00096068386
# train step:  440
# global_step:  440
# learning rate:  0.0009605961
# train step:  441
# global_step:  441
# learning rate:  0.00096050836
# train step:  442
# global_step:  442
# learning rate:  0.0009604206
# train step:  443
# global_step:  443
# learning rate:  0.00096033287
# train step:  444
# global_step:  444
# learning rate:  0.00096024515
# train step:  445
# global_step:  445
# learning rate:  0.00096015737
# train step:  446
# global_step:  446
# learning rate:  0.00096006965
# train step:  447
# global_step:  447
# learning rate:  0.00095998193
# train step:  448
# global_step:  448
# learning rate:  0.0009598942
# train step:  449
# global_step:  449
# learning rate:  0.00095980655
# train step:  450
# global_step:  450
# learning rate:  0.00095971883
# train step:  451
# global_step:  451
# learning rate:  0.0009596311
# train step:  452
# global_step:  452
# learning rate:  0.00095954345
# train step:  453
# global_step:  453
# learning rate:  0.00095945585
# train step:  454
# global_step:  454
# learning rate:  0.00095936813
# train step:  455
# global_step:  455
# learning rate:  0.00095928047
# train step:  456
# global_step:  456
# learning rate:  0.00095919287
# train step:  457
# global_step:  457
# learning rate:  0.00095910527
# train step:  458
# global_step:  458
# learning rate:  0.0009590176
# train step:  459
# global_step:  459
# learning rate:  0.00095893
# train step:  460
# global_step:  460
# learning rate:  0.0009588424
# train step:  461
# global_step:  461
# learning rate:  0.00095875474
# train step:  462
# global_step:  462
# learning rate:  0.00095866714
# train step:  463
# global_step:  463
# learning rate:  0.0009585796
# train step:  464
# global_step:  464
# learning rate:  0.00095849205
# train step:  465
# global_step:  465
# learning rate:  0.00095840445
# train step:  466
# global_step:  466
# learning rate:  0.00095831684
# train step:  467
# global_step:  467
# learning rate:  0.00095822936
# train step:  468
# global_step:  468
# learning rate:  0.0009581418
# train step:  469
# global_step:  469
# learning rate:  0.0009580542
# train step:  470
# global_step:  470
# learning rate:  0.0009579667
# train step:  471
# global_step:  471
# learning rate:  0.0009578792
# train step:  472
# global_step:  472
# learning rate:  0.0009577917
# train step:  473
# global_step:  473
# learning rate:  0.00095770415
# train step:  474
# global_step:  474
# learning rate:  0.00095761666
# train step:  475
# global_step:  475
# learning rate:  0.0009575292
# train step:  476
# global_step:  476
# learning rate:  0.0009574417
# train step:  477
# global_step:  477
# learning rate:  0.0009573542
# train step:  478
# global_step:  478
# learning rate:  0.0009572667
# train step:  479
# global_step:  479
# learning rate:  0.0009571793
# train step:  480
# global_step:  480
# learning rate:  0.00095709186
# train step:  481
# global_step:  481
# learning rate:  0.00095700443
# train step:  482
# global_step:  482
# learning rate:  0.00095691695
# train step:  483
# global_step:  483
# learning rate:  0.0009568295
# train step:  484
# global_step:  484
# learning rate:  0.0009567421
# train step:  485
# global_step:  485
# learning rate:  0.0009566547
# train step:  486
# global_step:  486
# learning rate:  0.00095656735
# train step:  487
# global_step:  487
# learning rate:  0.00095647987
# train step:  488
# global_step:  488
# learning rate:  0.0009563925
# train step:  489
# global_step:  489
# learning rate:  0.0009563051
# train step:  490
# global_step:  490
# learning rate:  0.00095621776
# train step:  491
# global_step:  491
# learning rate:  0.00095613045
# train step:  492
# global_step:  492
# learning rate:  0.0009560431
# train step:  493
# global_step:  493
# learning rate:  0.0009559557
# train step:  494
# global_step:  494
# learning rate:  0.0009558684
# train step:  495
# global_step:  495
# learning rate:  0.000955781
# train step:  496
# global_step:  496
# learning rate:  0.0009556937
# train step:  497
# global_step:  497
# learning rate:  0.0009556064
# train step:  498
# global_step:  498
# learning rate:  0.00095551915
# train step:  499
# global_step:  499
# learning rate:  0.0009554318
# train step:  500
# global_step:  500
# learning rate:  0.0009553445
# train step:  501
# global_step:  501
# learning rate:  0.0009552572
# train step:  502
# global_step:  502
# learning rate:  0.00095516996
# train step:  503
# global_step:  503
# learning rate:  0.0009550827
# train step:  504
# global_step:  504
# learning rate:  0.00095499546
# train step:  505
# global_step:  505
# learning rate:  0.0009549082
# train step:  506
# global_step:  506
# learning rate:  0.0009548209
# train step:  507
# global_step:  507
# learning rate:  0.0009547337
# train step:  508
# global_step:  508
# learning rate:  0.00095464644
# train step:  509
# global_step:  509
# learning rate:  0.00095455925
# train step:  510
# global_step:  510
# learning rate:  0.00095447205
# train step:  511
# global_step:  511
# learning rate:  0.00095438486
# train step:  512
# global_step:  512
# learning rate:  0.00095429766
# train step:  513
# global_step:  513
# learning rate:  0.00095421047
# train step:  514
# global_step:  514
# learning rate:  0.00095412333
# train step:  515
# global_step:  515
# learning rate:  0.0009540361
# train step:  516
# global_step:  516
# learning rate:  0.00095394894
# train step:  517
# global_step:  517
# learning rate:  0.0009538618
# train step:  518
# global_step:  518
# learning rate:  0.00095377467
# train step:  519
# global_step:  519
# learning rate:  0.00095368753
# train step:  520
# global_step:  520
# learning rate:  0.0009536004
# train step:  521
# global_step:  521
# learning rate:  0.00095351326
# train step:  522
# global_step:  522
# learning rate:  0.0009534262
# train step:  523
# global_step:  523
# learning rate:  0.00095333904
# train step:  524
# global_step:  524
# learning rate:  0.00095325196
# train step:  525
# global_step:  525
# learning rate:  0.0009531649
# train step:  526
# global_step:  526
# learning rate:  0.0009530778
# train step:  527
# global_step:  527
# learning rate:  0.00095299067
# train step:  528
# global_step:  528
# learning rate:  0.0009529036
# train step:  529
# global_step:  529
# learning rate:  0.0009528165
# train step:  530
# global_step:  530
# learning rate:  0.0009527295
# train step:  531
# global_step:  531
# learning rate:  0.00095264247
# train step:  532
# global_step:  532
# learning rate:  0.0009525554
# train step:  533
# global_step:  533
# learning rate:  0.00095246837
# train step:  534
# global_step:  534
# learning rate:  0.00095238135
# train step:  535
# global_step:  535
# learning rate:  0.00095229433
# train step:  536
# global_step:  536
# learning rate:  0.00095220737
# train step:  537
# global_step:  537
# learning rate:  0.00095212035
# train step:  538
# global_step:  538
# learning rate:  0.0009520334
# train step:  539
# global_step:  539
# learning rate:  0.0009519464
# train step:  540
# global_step:  540
# learning rate:  0.0009518594
# train step:  541
# global_step:  541
# learning rate:  0.00095177244
# train step:  542
# global_step:  542
# learning rate:  0.00095168554
# train step:  543
# global_step:  543
# learning rate:  0.0009515986
# train step:  544
# global_step:  544
# learning rate:  0.0009515116
# train step:  545
# global_step:  545
# learning rate:  0.0009514247
# train step:  546
# global_step:  546
# learning rate:  0.00095133774
# train step:  547
# global_step:  547
# learning rate:  0.00095125084
# train step:  548
# global_step:  548
# learning rate:  0.00095116394
# train step:  549
# global_step:  549
# learning rate:  0.00095107703
# train step:  550
# global_step:  550
# learning rate:  0.0009509901
# train step:  551
# global_step:  551
# learning rate:  0.0009509033
# train step:  552
# global_step:  552
# learning rate:  0.0009508164
# train step:  553
# global_step:  553
# learning rate:  0.00095072953
# train step:  554
# global_step:  554
# learning rate:  0.0009506426
# train step:  555
# global_step:  555
# learning rate:  0.0009505558
# train step:  556
# global_step:  556
# learning rate:  0.00095046894
# train step:  557
# global_step:  557
# learning rate:  0.0009503821
# train step:  558
# global_step:  558
# learning rate:  0.0009502953
# train step:  559
# global_step:  559
# learning rate:  0.00095020846
# train step:  560
# global_step:  560
# learning rate:  0.0009501216
# train step:  561
# global_step:  561
# learning rate:  0.0009500348
# train step:  562
# global_step:  562
# learning rate:  0.00094994804
# train step:  563
# global_step:  563
# learning rate:  0.00094986125
# train step:  564
# global_step:  564
# learning rate:  0.00094977446
# train step:  565
# global_step:  565
# learning rate:  0.00094968773
# train step:  566
# global_step:  566
# learning rate:  0.00094960094
# train step:  567
# global_step:  567
# learning rate:  0.0009495142
# train step:  568
# global_step:  568
# learning rate:  0.0009494274
# train step:  569
# global_step:  569
# learning rate:  0.0009493407
# train step:  570
# global_step:  570
# learning rate:  0.00094925397
# train step:  571
# global_step:  571
# learning rate:  0.00094916724
# train step:  572
# global_step:  572
# learning rate:  0.0009490805
# train step:  573
# global_step:  573
# learning rate:  0.0009489938
# train step:  574
# global_step:  574
# learning rate:  0.0009489071
# train step:  575
# global_step:  575
# learning rate:  0.0009488204
# train step:  576
# global_step:  576
# learning rate:  0.0009487337
# train step:  577
# global_step:  577
# learning rate:  0.0009486471
# train step:  578
# global_step:  578
# learning rate:  0.00094856037
# train step:  579
# global_step:  579
# learning rate:  0.0009484737
# train step:  580
# global_step:  580
# learning rate:  0.0009483871
# train step:  581
# global_step:  581
# learning rate:  0.0009483004
# train step:  582
# global_step:  582
# learning rate:  0.00094821374
# train step:  583
# global_step:  583
# learning rate:  0.0009481271
# train step:  584
# global_step:  584
# learning rate:  0.0009480405
# train step:  585
# global_step:  585
# learning rate:  0.00094795384
# train step:  586
# global_step:  586
# learning rate:  0.0009478673
# train step:  587
# global_step:  587
# learning rate:  0.00094778073
# train step:  588
# global_step:  588
# learning rate:  0.0009476941
# train step:  589
# global_step:  589
# learning rate:  0.0009476075
# train step:  590
# global_step:  590
# learning rate:  0.00094752095
# train step:  591
# global_step:  591
# learning rate:  0.00094743434
# train step:  592
# global_step:  592
# learning rate:  0.0009473478
# train step:  593
# global_step:  593
# learning rate:  0.0009472613
# train step:  594
# global_step:  594
# learning rate:  0.00094717473
# train step:  595
# global_step:  595
# learning rate:  0.0009470882
# train step:  596
# global_step:  596
# learning rate:  0.0009470016
# train step:  597
# global_step:  597
# learning rate:  0.0009469151
# train step:  598
# global_step:  598
# learning rate:  0.0009468286
# train step:  599
# global_step:  599
# learning rate:  0.0009467421
# train step:  600
# global_step:  600
# learning rate:  0.00094665564
# train step:  601
# global_step:  601
# learning rate:  0.00094656914
# train step:  602
# global_step:  602
# learning rate:  0.00094648264
# train step:  603
# global_step:  603
# learning rate:  0.00094639615
# train step:  604
# global_step:  604
# learning rate:  0.0009463097
# train step:  605
# global_step:  605
# learning rate:  0.00094622327
# train step:  606
# global_step:  606
# learning rate:  0.00094613683
# train step:  607
# global_step:  607
# learning rate:  0.00094605034
# train step:  608
# global_step:  608
# learning rate:  0.0009459639
# train step:  609
# global_step:  609
# learning rate:  0.00094587746
# train step:  610
# global_step:  610
# learning rate:  0.0009457911
# train step:  611
# global_step:  611
# learning rate:  0.0009457047
# train step:  612
# global_step:  612
# learning rate:  0.00094561826
# train step:  613
# global_step:  613
# learning rate:  0.0009455319
# train step:  614
# global_step:  614
# learning rate:  0.00094544544
# train step:  615
# global_step:  615
# learning rate:  0.0009453591
# train step:  616
# global_step:  616
# learning rate:  0.00094527274
# train step:  617
# global_step:  617
# learning rate:  0.00094518636
# train step:  618
# global_step:  618
# learning rate:  0.0009451
# train step:  619
# global_step:  619
# learning rate:  0.0009450137
# train step:  620
# global_step:  620
# learning rate:  0.00094492733
# train step:  621
# global_step:  621
# learning rate:  0.000944841
# train step:  622
# global_step:  622
# learning rate:  0.0009447547
# train step:  623
# global_step:  623
# learning rate:  0.00094466837
# train step:  624
# global_step:  624
# learning rate:  0.0009445821
# train step:  625
# global_step:  625
# learning rate:  0.0009444958
# train step:  626
# global_step:  626
# learning rate:  0.00094440946
# train step:  627
# global_step:  627
# learning rate:  0.00094432314
# train step:  628
# global_step:  628
# learning rate:  0.00094423693
# train step:  629
# global_step:  629
# learning rate:  0.00094415067
# train step:  630
# global_step:  630
# learning rate:  0.00094406435
# train step:  631
# global_step:  631
# learning rate:  0.00094397814
# train step:  632
# global_step:  632
# learning rate:  0.0009438919
# train step:  633
# global_step:  633
# learning rate:  0.0009438056
# train step:  634
# global_step:  634
# learning rate:  0.0009437194
# train step:  635
# global_step:  635
# learning rate:  0.0009436332
# train step:  636
# global_step:  636
# learning rate:  0.000943547
# train step:  637
# global_step:  637
# learning rate:  0.0009434608
# train step:  638
# global_step:  638
# learning rate:  0.0009433746
# train step:  639
# global_step:  639
# learning rate:  0.0009432884
# train step:  640
# global_step:  640
# learning rate:  0.00094320223
# train step:  641
# global_step:  641
# learning rate:  0.000943116
# train step:  642
# global_step:  642
# learning rate:  0.0009430299
# train step:  643
# global_step:  643
# learning rate:  0.00094294373
# train step:  644
# global_step:  644
# learning rate:  0.0009428575
# train step:  645
# global_step:  645
# learning rate:  0.00094277144
# train step:  646
# global_step:  646
# learning rate:  0.0009426853
# train step:  647
# global_step:  647
# learning rate:  0.00094259914
# train step:  648
# global_step:  648
# learning rate:  0.00094251306
# train step:  649
# global_step:  649
# learning rate:  0.00094242697
# train step:  650
# global_step:  650
# learning rate:  0.0009423408
# train step:  651
# global_step:  651
# learning rate:  0.0009422548
# train step:  652
# global_step:  652
# learning rate:  0.00094216864
# train step:  653
# global_step:  653
# learning rate:  0.00094208255
# train step:  654
# global_step:  654
# learning rate:  0.0009419965
# train step:  655
# global_step:  655
# learning rate:  0.00094191043
# train step:  656
# global_step:  656
# learning rate:  0.00094182434
# train step:  657
# global_step:  657
# learning rate:  0.00094173837
# train step:  658
# global_step:  658
# learning rate:  0.0009416523
# train step:  659
# global_step:  659
# learning rate:  0.00094156625
# train step:  660
# global_step:  660
# learning rate:  0.0009414803
# train step:  661
# global_step:  661
# learning rate:  0.00094139425
# train step:  662
# global_step:  662
# learning rate:  0.0009413083
# train step:  663
# global_step:  663
# learning rate:  0.00094122224
# train step:  664
# global_step:  664
# learning rate:  0.0009411362
# train step:  665
# global_step:  665
# learning rate:  0.0009410503
# train step:  666
# global_step:  666
# learning rate:  0.00094096427
# train step:  667
# global_step:  667
# learning rate:  0.0009408783
# train step:  668
# global_step:  668
# learning rate:  0.0009407924
# train step:  669
# global_step:  669
# learning rate:  0.0009407064
# train step:  670
# global_step:  670
# learning rate:  0.0009406205
# train step:  671
# global_step:  671
# learning rate:  0.0009405345
# train step:  672
# global_step:  672
# learning rate:  0.00094044855
# train step:  673
# global_step:  673
# learning rate:  0.0009403627
# train step:  674
# global_step:  674
# learning rate:  0.0009402767
# train step:  675
# global_step:  675
# learning rate:  0.00094019086
# train step:  676
# global_step:  676
# learning rate:  0.00094010495
# train step:  677
# global_step:  677
# learning rate:  0.00094001903
# train step:  678
# global_step:  678
# learning rate:  0.0009399332
# train step:  679
# global_step:  679
# learning rate:  0.00093984726
# train step:  680
# global_step:  680
# learning rate:  0.00093976146
# train step:  681
# global_step:  681
# learning rate:  0.00093967555
# train step:  682
# global_step:  682
# learning rate:  0.00093958975
# train step:  683
# global_step:  683
# learning rate:  0.0009395039
# train step:  684
# global_step:  684
# learning rate:  0.00093941804
# train step:  685
# global_step:  685
# learning rate:  0.00093933224
# train step:  686
# global_step:  686
# learning rate:  0.0009392464
# train step:  687
# global_step:  687
# learning rate:  0.0009391606
# train step:  688
# global_step:  688
# learning rate:  0.0009390747
# train step:  689
# global_step:  689
# learning rate:  0.000938989
# train step:  690
# global_step:  690
# learning rate:  0.0009389032
# train step:  691
# global_step:  691
# learning rate:  0.00093881745
# train step:  692
# global_step:  692
# learning rate:  0.00093873165
# train step:  693
# global_step:  693
# learning rate:  0.0009386459
# train step:  694
# global_step:  694
# learning rate:  0.0009385601
# train step:  695
# global_step:  695
# learning rate:  0.0009384743
# train step:  696
# global_step:  696
# learning rate:  0.00093838864
# train step:  697
# global_step:  697
# learning rate:  0.00093830284
# train step:  698
# global_step:  698
# learning rate:  0.00093821716
# train step:  699
# global_step:  699
# learning rate:  0.0009381314
# train step:  700
# global_step:  700
# learning rate:  0.00093804573
# train step:  701
# global_step:  701
# learning rate:  0.00093796
# train step:  702
# global_step:  702
# learning rate:  0.00093787437
# train step:  703
# global_step:  703
# learning rate:  0.00093778863
# train step:  704
# global_step:  704
# learning rate:  0.000937703
# train step:  705
# global_step:  705
# learning rate:  0.00093761727
# train step:  706
# global_step:  706
# learning rate:  0.00093753164
# train step:  707
# global_step:  707
# learning rate:  0.00093744596
# train step:  708
# global_step:  708
# learning rate:  0.00093736034
# train step:  709
# global_step:  709
# learning rate:  0.00093727466
# train step:  710
# global_step:  710
# learning rate:  0.0009371891
# train step:  711
# global_step:  711
# learning rate:  0.0009371034
# train step:  712
# global_step:  712
# learning rate:  0.00093701784
# train step:  713
# global_step:  713
# learning rate:  0.00093693216
# train step:  714
# global_step:  714
# learning rate:  0.0009368466
# train step:  715
# global_step:  715
# learning rate:  0.00093676103
# train step:  716
# global_step:  716
# learning rate:  0.0009366754
# train step:  717
# global_step:  717
# learning rate:  0.0009365899
# train step:  718
# global_step:  718
# learning rate:  0.0009365043
# train step:  719
# global_step:  719
# learning rate:  0.0009364188
# train step:  720
# global_step:  720
# learning rate:  0.00093633315
# train step:  721
# global_step:  721
# learning rate:  0.00093624764
# train step:  722
# global_step:  722
# learning rate:  0.0009361621
# train step:  723
# global_step:  723
# learning rate:  0.00093607657
# train step:  724
# global_step:  724
# learning rate:  0.00093599106
# train step:  725
# global_step:  725
# learning rate:  0.0009359055
# train step:  726
# global_step:  726
# learning rate:  0.00093582005
# train step:  727
# global_step:  727
# learning rate:  0.0009357345
# train step:  728
# global_step:  728
# learning rate:  0.00093564903
# train step:  729
# global_step:  729
# learning rate:  0.0009355635
# train step:  730
# global_step:  730
# learning rate:  0.0009354781
# train step:  731
# global_step:  731
# learning rate:  0.0009353926
# train step:  732
# global_step:  732
# learning rate:  0.0009353071
# train step:  733
# global_step:  733
# learning rate:  0.00093522173
# train step:  734
# global_step:  734
# learning rate:  0.0009351362
# train step:  735
# global_step:  735
# learning rate:  0.00093505083
# train step:  736
# global_step:  736
# learning rate:  0.00093496544
# train step:  737
# global_step:  737
# learning rate:  0.00093487994
# train step:  738
# global_step:  738
# learning rate:  0.00093479455
# train step:  739
# global_step:  739
# learning rate:  0.00093470915
# train step:  740
# global_step:  740
# learning rate:  0.00093462376
# train step:  741
# global_step:  741
# learning rate:  0.0009345384
# train step:  742
# global_step:  742
# learning rate:  0.000934453
# train step:  743
# global_step:  743
# learning rate:  0.0009343676
# train step:  744
# global_step:  744
# learning rate:  0.0009342822
# train step:  745
# global_step:  745
# learning rate:  0.00093419687
# train step:  746
# global_step:  746
# learning rate:  0.00093411154
# train step:  747
# global_step:  747
# learning rate:  0.00093402615
# train step:  748
# global_step:  748
# learning rate:  0.0009339409
# train step:  749
# global_step:  749
# learning rate:  0.00093385554
# train step:  750
# global_step:  750
# learning rate:  0.0009337702
# train step:  751
# global_step:  751
# learning rate:  0.0009336849
# train step:  752
# global_step:  752
# learning rate:  0.0009335996
# train step:  753
# global_step:  753
# learning rate:  0.00093351427
# train step:  754
# global_step:  754
# learning rate:  0.000933429
# train step:  755
# global_step:  755
# learning rate:  0.0009333438
# train step:  756
# global_step:  756
# learning rate:  0.00093325845
# train step:  757
# global_step:  757
# learning rate:  0.00093317317
# train step:  758
# global_step:  758
# learning rate:  0.00093308795
# train step:  759
# global_step:  759
# learning rate:  0.0009330027
# train step:  760
# global_step:  760
# learning rate:  0.00093291746
# train step:  761
# global_step:  761
# learning rate:  0.00093283225
# train step:  762
# global_step:  762
# learning rate:  0.000932747
# train step:  763
# global_step:  763
# learning rate:  0.00093266176
# train step:  764
# global_step:  764
# learning rate:  0.0009325766
# train step:  765
# global_step:  765
# learning rate:  0.0009324913
# train step:  766
# global_step:  766
# learning rate:  0.00093240617
# train step:  767
# global_step:  767
# learning rate:  0.000932321
# train step:  768
# global_step:  768
# learning rate:  0.00093223574
# train step:  769
# global_step:  769
# learning rate:  0.00093215064
# train step:  770
# global_step:  770
# learning rate:  0.0009320655
# train step:  771
# global_step:  771
# learning rate:  0.0009319803
# train step:  772
# global_step:  772
# learning rate:  0.00093189516
# train step:  773
# global_step:  773
# learning rate:  0.00093181
# train step:  774
# global_step:  774
# learning rate:  0.0009317249
# train step:  775
# global_step:  775
# learning rate:  0.00093163975
# train step:  776
# global_step:  776
# learning rate:  0.00093155465
# train step:  777
# global_step:  777
# learning rate:  0.00093146955
# train step:  778
# global_step:  778
# learning rate:  0.00093138445
# train step:  779
# global_step:  779
# learning rate:  0.0009312993
# train step:  780
# global_step:  780
# learning rate:  0.00093121425
# train step:  781
# global_step:  781
# learning rate:  0.00093112915
# train step:  782
# global_step:  782
# learning rate:  0.0009310441
# train step:  783
# global_step:  783
# learning rate:  0.000930959
# train step:  784
# global_step:  784
# learning rate:  0.00093087397
# train step:  785
# global_step:  785
# learning rate:  0.0009307889
# train step:  786
# global_step:  786
# learning rate:  0.00093070394
# train step:  787
# global_step:  787
# learning rate:  0.00093061884
# train step:  788
# global_step:  788
# learning rate:  0.00093053386
# train step:  789
# global_step:  789
# learning rate:  0.0009304488
# train step:  790
# global_step:  790
# learning rate:  0.00093036384
# train step:  791
# global_step:  791
# learning rate:  0.0009302788
# train step:  792
# global_step:  792
# learning rate:  0.0009301938
# train step:  793
# global_step:  793
# learning rate:  0.0009301088
# train step:  794
# global_step:  794
# learning rate:  0.0009300239
# train step:  795
# global_step:  795
# learning rate:  0.0009299389
# train step:  796
# global_step:  796
# learning rate:  0.00092985394
# train step:  797
# global_step:  797
# learning rate:  0.00092976895
# train step:  798
# global_step:  798
# learning rate:  0.000929684
# train step:  799
# global_step:  799
# learning rate:  0.0009295991
# train step:  800
# global_step:  800
# learning rate:  0.0009295142
# train step:  801
# global_step:  801
# learning rate:  0.0009294292
# train step:  802
# global_step:  802
# learning rate:  0.0009293443
# train step:  803
# global_step:  803
# learning rate:  0.0009292594
# train step:  804
# global_step:  804
# learning rate:  0.00092917454
# train step:  805
# global_step:  805
# learning rate:  0.00092908967
# train step:  806
# global_step:  806
# learning rate:  0.0009290047
# train step:  807
# global_step:  807
# learning rate:  0.0009289199
# train step:  808
# global_step:  808
# learning rate:  0.000928835
# train step:  809
# global_step:  809
# learning rate:  0.00092875015
# train step:  810
# global_step:  810
# learning rate:  0.0009286653
# train step:  811
# global_step:  811
# learning rate:  0.00092858047
# train step:  812
# global_step:  812
# learning rate:  0.0009284956
# train step:  813
# global_step:  813
# learning rate:  0.00092841074
# train step:  814
# global_step:  814
# learning rate:  0.0009283259
# train step:  815
# global_step:  815
# learning rate:  0.0009282411
# train step:  816
# global_step:  816
# learning rate:  0.00092815637
# train step:  817
# global_step:  817
# learning rate:  0.00092807156
# train step:  818
# global_step:  818
# learning rate:  0.00092798675
# train step:  819
# global_step:  819
# learning rate:  0.00092790194
# train step:  820
# global_step:  820
# learning rate:  0.0009278172
# train step:  821
# global_step:  821
# learning rate:  0.0009277324
# train step:  822
# global_step:  822
# learning rate:  0.00092764763
# train step:  823
# global_step:  823
# learning rate:  0.0009275629
# train step:  824
# global_step:  824
# learning rate:  0.0009274782
# train step:  825
# global_step:  825
# learning rate:  0.00092739344
# train step:  826
# global_step:  826
# learning rate:  0.00092730875
# train step:  827
# global_step:  827
# learning rate:  0.00092722394
# train step:  828
# global_step:  828
# learning rate:  0.00092713925
# train step:  829
# global_step:  829
# learning rate:  0.00092705456
# train step:  830
# global_step:  830
# learning rate:  0.00092696986
# train step:  831
# global_step:  831
# learning rate:  0.0009268852
# train step:  832
# global_step:  832
# learning rate:  0.0009268005
# train step:  833
# global_step:  833
# learning rate:  0.00092671585
# train step:  834
# global_step:  834
# learning rate:  0.00092663115
# train step:  835
# global_step:  835
# learning rate:  0.0009265465
# train step:  836
# global_step:  836
# learning rate:  0.0009264619
# train step:  837
# global_step:  837
# learning rate:  0.00092637714
# train step:  838
# global_step:  838
# learning rate:  0.0009262925
# train step:  839
# global_step:  839
# learning rate:  0.0009262079
# train step:  840
# global_step:  840
# learning rate:  0.0009261233
# train step:  841
# global_step:  841
# learning rate:  0.00092603866
# train step:  842
# global_step:  842
# learning rate:  0.0009259541
# train step:  843
# global_step:  843
# learning rate:  0.0009258695
# train step:  844
# global_step:  844
# learning rate:  0.0009257849
# train step:  845
# global_step:  845
# learning rate:  0.0009257003
# train step:  846
# global_step:  846
# learning rate:  0.0009256157
# train step:  847
# global_step:  847
# learning rate:  0.0009255312
# train step:  848
# global_step:  848
# learning rate:  0.0009254466
# train step:  849
# global_step:  849
# learning rate:  0.0009253621
# train step:  850
# global_step:  850
# learning rate:  0.00092527753
# train step:  851
# global_step:  851
# learning rate:  0.000925193
# train step:  852
# global_step:  852
# learning rate:  0.0009251085
# train step:  853
# global_step:  853
# learning rate:  0.000925024
# train step:  854
# global_step:  854
# learning rate:  0.00092493946
# train step:  855
# global_step:  855
# learning rate:  0.00092485495
# train step:  856
# global_step:  856
# learning rate:  0.0009247704
# train step:  857
# global_step:  857
# learning rate:  0.0009246859
# train step:  858
# global_step:  858
# learning rate:  0.00092460145
# train step:  859
# global_step:  859
# learning rate:  0.00092451693
# train step:  860
# global_step:  860
# learning rate:  0.0009244325
# train step:  861
# global_step:  861
# learning rate:  0.000924348
# train step:  862
# global_step:  862
# learning rate:  0.00092426356
# train step:  863
# global_step:  863
# learning rate:  0.00092417916
# train step:  864
# global_step:  864
# learning rate:  0.0009240947
# train step:  865
# global_step:  865
# learning rate:  0.0009240103
# train step:  866
# global_step:  866
# learning rate:  0.00092392584
# train step:  867
# global_step:  867
# learning rate:  0.00092384143
# train step:  868
# global_step:  868
# learning rate:  0.00092375703
# train step:  869
# global_step:  869
# learning rate:  0.00092367263
# train step:  870
# global_step:  870
# learning rate:  0.00092358823
# train step:  871
# global_step:  871
# learning rate:  0.00092350395
# train step:  872
# global_step:  872
# learning rate:  0.00092341955
# train step:  873
# global_step:  873
# learning rate:  0.0009233352
# train step:  874
# global_step:  874
# learning rate:  0.0009232508
# train step:  875
# global_step:  875
# learning rate:  0.00092316646
# train step:  876
# global_step:  876
# learning rate:  0.0009230821
# train step:  877
# global_step:  877
# learning rate:  0.0009229978
# train step:  878
# global_step:  878
# learning rate:  0.00092291343
# train step:  879
# global_step:  879
# learning rate:  0.00092282915
# train step:  880
# global_step:  880
# learning rate:  0.0009227448
# train step:  881
# global_step:  881
# learning rate:  0.0009226605
# train step:  882
# global_step:  882
# learning rate:  0.00092257623
# train step:  883
# global_step:  883
# learning rate:  0.0009224919
# train step:  884
# global_step:  884
# learning rate:  0.0009224076
# train step:  885
# global_step:  885
# learning rate:  0.0009223234
# train step:  886
# global_step:  886
# learning rate:  0.0009222391
# train step:  887
# global_step:  887
# learning rate:  0.0009221548
# train step:  888
# global_step:  888
# learning rate:  0.0009220706
# train step:  889
# global_step:  889
# learning rate:  0.0009219863
# train step:  890
# global_step:  890
# learning rate:  0.00092190213
# train step:  891
# global_step:  891
# learning rate:  0.0009218179
# train step:  892
# global_step:  892
# learning rate:  0.0009217337
# train step:  893
# global_step:  893
# learning rate:  0.00092164945
# train step:  894
# global_step:  894
# learning rate:  0.0009215653
# train step:  895
# global_step:  895
# learning rate:  0.00092148106
# train step:  896
# global_step:  896
# learning rate:  0.0009213969
# train step:  897
# global_step:  897
# learning rate:  0.00092131266
# train step:  898
# global_step:  898
# learning rate:  0.0009212285
# train step:  899
# global_step:  899
# learning rate:  0.0009211443
# train step:  900
# global_step:  900
# learning rate:  0.0009210602
# train step:  901
# global_step:  901
# learning rate:  0.00092097605
# train step:  902
# global_step:  902
# learning rate:  0.00092089194
# train step:  903
# global_step:  903
# learning rate:  0.00092080777
# train step:  904
# global_step:  904
# learning rate:  0.00092072366
# train step:  905
# global_step:  905
# learning rate:  0.0009206395
# train step:  906
# global_step:  906
# learning rate:  0.0009205554
# train step:  907
# global_step:  907
# learning rate:  0.00092047127
# train step:  908
# global_step:  908
# learning rate:  0.0009203872
# train step:  909
# global_step:  909
# learning rate:  0.0009203031
# train step:  910
# global_step:  910
# learning rate:  0.00092021906
# train step:  911
# global_step:  911
# learning rate:  0.00092013495
# train step:  912
# global_step:  912
# learning rate:  0.0009200509
# train step:  913
# global_step:  913
# learning rate:  0.00091996684
# train step:  914
# global_step:  914
# learning rate:  0.00091988273
# train step:  915
# global_step:  915
# learning rate:  0.00091979874
# train step:  916
# global_step:  916
# learning rate:  0.00091971474
# train step:  917
# global_step:  917
# learning rate:  0.0009196307
# train step:  918
# global_step:  918
# learning rate:  0.00091954664
# train step:  919
# global_step:  919
# learning rate:  0.00091946265
# train step:  920
# global_step:  920
# learning rate:  0.0009193786
# train step:  921
# global_step:  921
# learning rate:  0.00091929466
# train step:  922
# global_step:  922
# learning rate:  0.00091921067
# train step:  923
# global_step:  923
# learning rate:  0.0009191267
# train step:  924
# global_step:  924
# learning rate:  0.0009190427
# train step:  925
# global_step:  925
# learning rate:  0.0009189587
# train step:  926
# global_step:  926
# learning rate:  0.0009188748
# train step:  927
# global_step:  927
# learning rate:  0.0009187908
# train step:  928
# global_step:  928
# learning rate:  0.0009187069
# train step:  929
# global_step:  929
# learning rate:  0.00091862294
# train step:  930
# global_step:  930
# learning rate:  0.000918539
# train step:  931
# global_step:  931
# learning rate:  0.00091845513
# train step:  932
# global_step:  932
# learning rate:  0.0009183712
# train step:  933
# global_step:  933
# learning rate:  0.00091828726
# train step:  934
# global_step:  934
# learning rate:  0.0009182034
# train step:  935
# global_step:  935
# learning rate:  0.00091811945
# train step:  936
# global_step:  936
# learning rate:  0.0009180356
# train step:  937
# global_step:  937
# learning rate:  0.00091795175
# train step:  938
# global_step:  938
# learning rate:  0.0009178679
# train step:  939
# global_step:  939
# learning rate:  0.000917784
# train step:  940
# global_step:  940
# learning rate:  0.0009177002
# train step:  941
# global_step:  941
# learning rate:  0.00091761636
# train step:  942
# global_step:  942
# learning rate:  0.0009175325
# train step:  943
# global_step:  943
# learning rate:  0.00091744866
# train step:  944
# global_step:  944
# learning rate:  0.00091736484
# train step:  945
# global_step:  945
# learning rate:  0.000917281
# train step:  946
# global_step:  946
# learning rate:  0.0009171972
# train step:  947
# global_step:  947
# learning rate:  0.0009171134
# train step:  948
# global_step:  948
# learning rate:  0.0009170296
# train step:  949
# global_step:  949
# learning rate:  0.00091694586
# train step:  950
# global_step:  950
# learning rate:  0.00091686205
# train step:  951
# global_step:  951
# learning rate:  0.0009167783
# train step:  952
# global_step:  952
# learning rate:  0.0009166946
# train step:  953
# global_step:  953
# learning rate:  0.0009166108
# train step:  954
# global_step:  954
# learning rate:  0.00091652706
# train step:  955
# global_step:  955
# learning rate:  0.00091644336
# train step:  956
# global_step:  956
# learning rate:  0.0009163596
# train step:  957
# global_step:  957
# learning rate:  0.00091627584
# train step:  958
# global_step:  958
# learning rate:  0.00091619213
# train step:  959
# global_step:  959
# learning rate:  0.0009161085
# train step:  960
# global_step:  960
# learning rate:  0.0009160247
# train step:  961
# global_step:  961
# learning rate:  0.000915941
# train step:  962
# global_step:  962
# learning rate:  0.0009158574
# train step:  963
# global_step:  963
# learning rate:  0.00091577374
# train step:  964
# global_step:  964
# learning rate:  0.00091569003
# train step:  965
# global_step:  965
# learning rate:  0.0009156064
# train step:  966
# global_step:  966
# learning rate:  0.00091552275
# train step:  967
# global_step:  967
# learning rate:  0.0009154391
# train step:  968
# global_step:  968
# learning rate:  0.00091535546
# train step:  969
# global_step:  969
# learning rate:  0.0009152718
# train step:  970
# global_step:  970
# learning rate:  0.00091518817
# train step:  971
# global_step:  971
# learning rate:  0.00091510464
# train step:  972
# global_step:  972
# learning rate:  0.000915021
# train step:  973
# global_step:  973
# learning rate:  0.00091493735
# train step:  974
# global_step:  974
# learning rate:  0.0009148538
# train step:  975
# global_step:  975
# learning rate:  0.00091477024
# train step:  976
# global_step:  976
# learning rate:  0.0009146866
# train step:  977
# global_step:  977
# learning rate:  0.00091460306
# train step:  978
# global_step:  978
# learning rate:  0.0009145195
# train step:  979
# global_step:  979
# learning rate:  0.00091443595
# train step:  980
# global_step:  980
# learning rate:  0.0009143524
# train step:  981
# global_step:  981
# learning rate:  0.0009142689
# train step:  982
# global_step:  982
# learning rate:  0.0009141853
# train step:  983
# global_step:  983
# learning rate:  0.00091410184
# train step:  984
# global_step:  984
# learning rate:  0.0009140183
# train step:  985
# global_step:  985
# learning rate:  0.0009139348
# train step:  986
# global_step:  986
# learning rate:  0.0009138513
# train step:  987
# global_step:  987
# learning rate:  0.0009137678
# train step:  988
# global_step:  988
# learning rate:  0.0009136843
# train step:  989
# global_step:  989
# learning rate:  0.00091360084
# train step:  990
# global_step:  990
# learning rate:  0.0009135174
# train step:  991
# global_step:  991
# learning rate:  0.0009134339
# train step:  992
# global_step:  992
# learning rate:  0.00091335044
# train step:  993
# global_step:  993
# learning rate:  0.000913267
# train step:  994
# global_step:  994
# learning rate:  0.00091318355
# train step:  995
# global_step:  995
# learning rate:  0.0009131001
# train step:  996
# global_step:  996
# learning rate:  0.00091301673
# train step:  997
# global_step:  997
# learning rate:  0.00091293326
# train step:  998
# global_step:  998
# learning rate:  0.0009128499
# train step:  999
# global_step:  999
# learning rate:  0.0009127665
# train step:  1000
# global_step:  1000
# learning rate:  0.0009126831
# train step:  1001
# global_step:  1001
# learning rate:  0.00091259973
# train step:  1002
# global_step:  1002
# learning rate:  0.0009125163
# train step:  1003
# global_step:  1003
# learning rate:  0.00091243297
# train step:  1004
# global_step:  1004
# learning rate:  0.0009123496
# train step:  1005
# global_step:  1005
# learning rate:  0.00091226626
# train step:  1006
# global_step:  1006
# learning rate:  0.0009121829
# train step:  1007
# global_step:  1007
# learning rate:  0.0009120996
# train step:  1008
# global_step:  1008
# learning rate:  0.00091201626
# train step:  1009
# global_step:  1009
# learning rate:  0.0009119329
# train step:  1010
# global_step:  1010
# learning rate:  0.0009118496
# train step:  1011
# global_step:  1011
# learning rate:  0.00091176626
# train step:  1012
# global_step:  1012
# learning rate:  0.000911683
# train step:  1013
# global_step:  1013
# learning rate:  0.00091159967
# train step:  1014
# global_step:  1014
# learning rate:  0.0009115164
# train step:  1015
# global_step:  1015
# learning rate:  0.00091143313
# train step:  1016
# global_step:  1016
# learning rate:  0.0009113499
# train step:  1017
# global_step:  1017
# learning rate:  0.0009112666
# train step:  1018
# global_step:  1018
# learning rate:  0.00091118336
# train step:  1019
# global_step:  1019
# learning rate:  0.00091110007
# train step:  1020
# global_step:  1020
# learning rate:  0.0009110169
# train step:  1021
# global_step:  1021
# learning rate:  0.0009109336
# train step:  1022
# global_step:  1022
# learning rate:  0.0009108504
# train step:  1023
# global_step:  1023
# learning rate:  0.0009107672
# train step:  1024
# global_step:  1024
# learning rate:  0.000910684
# train step:  1025
# global_step:  1025
# learning rate:  0.00091060076
# train step:  1026
# global_step:  1026
# learning rate:  0.0009105176
# train step:  1027
# global_step:  1027
# learning rate:  0.00091043435
# train step:  1028
# global_step:  1028
# learning rate:  0.0009103512
# train step:  1029
# global_step:  1029
# learning rate:  0.00091026805
# train step:  1030
# global_step:  1030
# learning rate:  0.00091018487
# train step:  1031
# global_step:  1031
# learning rate:  0.0009101017
# train step:  1032
# global_step:  1032
# learning rate:  0.00091001857
# train step:  1033
# global_step:  1033
# learning rate:  0.0009099354
# train step:  1034
# global_step:  1034
# learning rate:  0.00090985233
# train step:  1035
# global_step:  1035
# learning rate:  0.00090976915
# train step:  1036
# global_step:  1036
# learning rate:  0.00090968603
# train step:  1037
# global_step:  1037
# learning rate:  0.0009096029
# train step:  1038
# global_step:  1038
# learning rate:  0.00090951985
# train step:  1039
# global_step:  1039
# learning rate:  0.0009094367
# train step:  1040
# global_step:  1040
# learning rate:  0.00090935366
# train step:  1041
# global_step:  1041
# learning rate:  0.00090927054
# train step:  1042
# global_step:  1042
# learning rate:  0.0009091875
# train step:  1043
# global_step:  1043
# learning rate:  0.0009091044
# train step:  1044
# global_step:  1044
# learning rate:  0.00090902136
# train step:  1045
# global_step:  1045
# learning rate:  0.00090893835
# train step:  1046
# global_step:  1046
# learning rate:  0.00090885523
# train step:  1047
# global_step:  1047
# learning rate:  0.0009087722
# train step:  1048
# global_step:  1048
# learning rate:  0.00090868917
# train step:  1049
# global_step:  1049
# learning rate:  0.0009086062
# train step:  1050
# global_step:  1050
# learning rate:  0.00090852316
# train step:  1051
# global_step:  1051
# learning rate:  0.00090844015
# train step:  1052
# global_step:  1052
# learning rate:  0.0009083572
# train step:  1053
# global_step:  1053
# learning rate:  0.00090827415
# train step:  1054
# global_step:  1054
# learning rate:  0.0009081912
# train step:  1055
# global_step:  1055
# learning rate:  0.0009081082
# train step:  1056
# global_step:  1056
# learning rate:  0.00090802525
# train step:  1057
# global_step:  1057
# learning rate:  0.0009079423
# train step:  1058
# global_step:  1058
# learning rate:  0.00090785936
# train step:  1059
# global_step:  1059
# learning rate:  0.0009077764
# train step:  1060
# global_step:  1060
# learning rate:  0.00090769347
# train step:  1061
# global_step:  1061
# learning rate:  0.0009076105
# train step:  1062
# global_step:  1062
# learning rate:  0.00090752763
# train step:  1063
# global_step:  1063
# learning rate:  0.0009074447
# train step:  1064
# global_step:  1064
# learning rate:  0.0009073618
# train step:  1065
# global_step:  1065
# learning rate:  0.00090727885
# train step:  1066
# global_step:  1066
# learning rate:  0.000907196
# train step:  1067
# global_step:  1067
# learning rate:  0.00090711314
# train step:  1068
# global_step:  1068
# learning rate:  0.00090703025
# train step:  1069
# global_step:  1069
# learning rate:  0.00090694736
# train step:  1070
# global_step:  1070
# learning rate:  0.00090686453
# train step:  1071
# global_step:  1071
# learning rate:  0.00090678164
# train step:  1072
# global_step:  1072
# learning rate:  0.0009066988
# train step:  1073
# global_step:  1073
# learning rate:  0.000906616
# train step:  1074
# global_step:  1074
# learning rate:  0.00090653315
# train step:  1075
# global_step:  1075
# learning rate:  0.0009064503
# train step:  1076
# global_step:  1076
# learning rate:  0.0009063675
# train step:  1077
# global_step:  1077
# learning rate:  0.0009062847
# train step:  1078
# global_step:  1078
# learning rate:  0.0009062019
# train step:  1079
# global_step:  1079
# learning rate:  0.00090611906
# train step:  1080
# global_step:  1080
# learning rate:  0.0009060363
# train step:  1081
# global_step:  1081
# learning rate:  0.0009059536
# train step:  1082
# global_step:  1082
# learning rate:  0.00090587075
# train step:  1083
# global_step:  1083
# learning rate:  0.000905788
# train step:  1084
# global_step:  1084
# learning rate:  0.00090570527
# train step:  1085
# global_step:  1085
# learning rate:  0.00090562255
# train step:  1086
# global_step:  1086
# learning rate:  0.0009055398
# train step:  1087
# global_step:  1087
# learning rate:  0.00090545707
# train step:  1088
# global_step:  1088
# learning rate:  0.00090537436
# train step:  1089
# global_step:  1089
# learning rate:  0.0009052916
# train step:  1090
# global_step:  1090
# learning rate:  0.0009052089
# train step:  1091
# global_step:  1091
# learning rate:  0.0009051262
# train step:  1092
# global_step:  1092
# learning rate:  0.00090504345
# train step:  1093
# global_step:  1093
# learning rate:  0.0009049608
# train step:  1094
# global_step:  1094
# learning rate:  0.00090487814
# train step:  1095
# global_step:  1095
# learning rate:  0.0009047954
# train step:  1096
# global_step:  1096
# learning rate:  0.00090471277
# train step:  1097
# global_step:  1097
# learning rate:  0.0009046301
# train step:  1098
# global_step:  1098
# learning rate:  0.0009045475
# train step:  1099
# global_step:  1099
# learning rate:  0.0009044648
# train step:  1100
# global_step:  1100
# learning rate:  0.0009043822
# train step:  1101
# global_step:  1101
# learning rate:  0.0009042996
# train step:  1102
# global_step:  1102
# learning rate:  0.000904217
# train step:  1103
# global_step:  1103
# learning rate:  0.0009041343
# train step:  1104
# global_step:  1104
# learning rate:  0.00090405176
# train step:  1105
# global_step:  1105
# learning rate:  0.00090396916
# train step:  1106
# global_step:  1106
# learning rate:  0.00090388657
# train step:  1107
# global_step:  1107
# learning rate:  0.000903804
# train step:  1108
# global_step:  1108
# learning rate:  0.0009037214
# train step:  1109
# global_step:  1109
# learning rate:  0.00090363884
# train step:  1110
# global_step:  1110
# learning rate:  0.0009035563
# train step:  1111
# global_step:  1111
# learning rate:  0.0009034737
# train step:  1112
# global_step:  1112
# learning rate:  0.00090339116
# train step:  1113
# global_step:  1113
# learning rate:  0.0009033087
# train step:  1114
# global_step:  1114
# learning rate:  0.00090322614
# train step:  1115
# global_step:  1115
# learning rate:  0.00090314355
# train step:  1116
# global_step:  1116
# learning rate:  0.00090306107
# train step:  1117
# global_step:  1117
# learning rate:  0.0009029786
# train step:  1118
# global_step:  1118
# learning rate:  0.0009028961
# train step:  1119
# global_step:  1119
# learning rate:  0.00090281357
# train step:  1120
# global_step:  1120
# learning rate:  0.0009027311
# train step:  1121
# global_step:  1121
# learning rate:  0.0009026486
# train step:  1122
# global_step:  1122
# learning rate:  0.0009025662
# train step:  1123
# global_step:  1123
# learning rate:  0.0009024837
# train step:  1124
# global_step:  1124
# learning rate:  0.0009024012
# train step:  1125
# global_step:  1125
# learning rate:  0.0009023188
# train step:  1126
# global_step:  1126
# learning rate:  0.0009022364
# train step:  1127
# global_step:  1127
# learning rate:  0.00090215396
# train step:  1128
# global_step:  1128
# learning rate:  0.00090207154
# train step:  1129
# global_step:  1129
# learning rate:  0.00090198906
# train step:  1130
# global_step:  1130
# learning rate:  0.0009019067
# train step:  1131
# global_step:  1131
# learning rate:  0.00090182427
# train step:  1132
# global_step:  1132
# learning rate:  0.0009017419
# train step:  1133
# global_step:  1133
# learning rate:  0.00090165954
# train step:  1134
# global_step:  1134
# learning rate:  0.0009015772
# train step:  1135
# global_step:  1135
# learning rate:  0.00090149476
# train step:  1136
# global_step:  1136
# learning rate:  0.0009014124
# train step:  1137
# global_step:  1137
# learning rate:  0.00090133003
# train step:  1138
# global_step:  1138
# learning rate:  0.0009012477
# train step:  1139
# global_step:  1139
# learning rate:  0.00090116536
# train step:  1140
# global_step:  1140
# learning rate:  0.00090108305
# train step:  1141
# global_step:  1141
# learning rate:  0.00090100075
# train step:  1142
# global_step:  1142
# learning rate:  0.0009009184
# train step:  1143
# global_step:  1143
# learning rate:  0.0009008361
# train step:  1144
# global_step:  1144
# learning rate:  0.0009007538
# train step:  1145
# global_step:  1145
# learning rate:  0.00090067147
# train step:  1146
# global_step:  1146
# learning rate:  0.0009005892
# train step:  1147
# global_step:  1147
# learning rate:  0.0009005069
# train step:  1148
# global_step:  1148
# learning rate:  0.00090042467
# train step:  1149
# global_step:  1149
# learning rate:  0.00090034236
# train step:  1150
# global_step:  1150
# learning rate:  0.0009002601
# train step:  1151
# global_step:  1151
# learning rate:  0.00090017787
# train step:  1152
# global_step:  1152
# learning rate:  0.0009000956
# train step:  1153
# global_step:  1153
# learning rate:  0.0009000134
# train step:  1154
# global_step:  1154
# learning rate:  0.0008999312
# train step:  1155
# global_step:  1155
# learning rate:  0.00089984894
# train step:  1156
# global_step:  1156
# learning rate:  0.00089976675
# train step:  1157
# global_step:  1157
# learning rate:  0.00089968456
# train step:  1158
# global_step:  1158
# learning rate:  0.0008996023
# train step:  1159
# global_step:  1159
# learning rate:  0.0008995201
# train step:  1160
# global_step:  1160
# learning rate:  0.00089943793
# train step:  1161
# global_step:  1161
# learning rate:  0.00089935574
# train step:  1162
# global_step:  1162
# learning rate:  0.0008992736
# train step:  1163
# global_step:  1163
# learning rate:  0.0008991914
# train step:  1164
# global_step:  1164
# learning rate:  0.0008991093
# train step:  1165
# global_step:  1165
# learning rate:  0.00089902716
# train step:  1166
# global_step:  1166
# learning rate:  0.000898945
# train step:  1167
# global_step:  1167
# learning rate:  0.0008988629
# train step:  1168
# global_step:  1168
# learning rate:  0.00089878077
# train step:  1169
# global_step:  1169
# learning rate:  0.0008986987
# train step:  1170
# global_step:  1170
# learning rate:  0.00089861656
# train step:  1171
# global_step:  1171
# learning rate:  0.00089853443
# train step:  1172
# global_step:  1172
# learning rate:  0.0008984523
# train step:  1173
# global_step:  1173
# learning rate:  0.0008983702
# train step:  1174
# global_step:  1174
# learning rate:  0.00089828816
# train step:  1175
# global_step:  1175
# learning rate:  0.0008982061
# train step:  1176
# global_step:  1176
# learning rate:  0.000898124
# train step:  1177
# global_step:  1177
# learning rate:  0.000898042
# train step:  1178
# global_step:  1178
# learning rate:  0.0008979599
# train step:  1179
# global_step:  1179
# learning rate:  0.0008978779
# train step:  1180
# global_step:  1180
# learning rate:  0.0008977959
# train step:  1181
# global_step:  1181
# learning rate:  0.0008977139
# train step:  1182
# global_step:  1182
# learning rate:  0.0008976318
# train step:  1183
# global_step:  1183
# learning rate:  0.00089754985
# train step:  1184
# global_step:  1184
# learning rate:  0.00089746783
# train step:  1185
# global_step:  1185
# learning rate:  0.0008973858
# train step:  1186
# global_step:  1186
# learning rate:  0.00089730386
# train step:  1187
# global_step:  1187
# learning rate:  0.00089722185
# train step:  1188
# global_step:  1188
# learning rate:  0.0008971399
# train step:  1189
# global_step:  1189
# learning rate:  0.00089705794
# train step:  1190
# global_step:  1190
# learning rate:  0.000896976
# train step:  1191
# global_step:  1191
# learning rate:  0.000896894
# train step:  1192
# global_step:  1192
# learning rate:  0.00089681207
# train step:  1193
# global_step:  1193
# learning rate:  0.00089673017
# train step:  1194
# global_step:  1194
# learning rate:  0.0008966482
# train step:  1195
# global_step:  1195
# learning rate:  0.0008965663
# train step:  1196
# global_step:  1196
# learning rate:  0.00089648436
# train step:  1197
# global_step:  1197
# learning rate:  0.00089640246
# train step:  1198
# global_step:  1198
# learning rate:  0.00089632056
# train step:  1199
# global_step:  1199
# learning rate:  0.00089623866
# train step:  1200
# global_step:  1200
# learning rate:  0.0008961568
# train step:  1201
# global_step:  1201
# learning rate:  0.0008960749
# train step:  1202
# global_step:  1202
# learning rate:  0.000895993
# train step:  1203
# global_step:  1203
# learning rate:  0.0008959112
# train step:  1204
# global_step:  1204
# learning rate:  0.00089582935
# train step:  1205
# global_step:  1205
# learning rate:  0.0008957475
# train step:  1206
# global_step:  1206
# learning rate:  0.00089566567
# train step:  1207
# global_step:  1207
# learning rate:  0.0008955838
# train step:  1208
# global_step:  1208
# learning rate:  0.000895502
# train step:  1209
# global_step:  1209
# learning rate:  0.00089542015
# train step:  1210
# global_step:  1210
# learning rate:  0.0008953384
# train step:  1211
# global_step:  1211
# learning rate:  0.0008952566
# train step:  1212
# global_step:  1212
# learning rate:  0.0008951748
# train step:  1213
# global_step:  1213
# learning rate:  0.000895093
# train step:  1214
# global_step:  1214
# learning rate:  0.00089501124
# train step:  1215
# global_step:  1215
# learning rate:  0.00089492946
# train step:  1216
# global_step:  1216
# learning rate:  0.0008948477
# train step:  1217
# global_step:  1217
# learning rate:  0.00089476595
# train step:  1218
# global_step:  1218
# learning rate:  0.00089468417
# train step:  1219
# global_step:  1219
# learning rate:  0.00089460245
# train step:  1220
# global_step:  1220
# learning rate:  0.0008945207
# train step:  1221
# global_step:  1221
# learning rate:  0.000894439
# train step:  1222
# global_step:  1222
# learning rate:  0.00089435733
# train step:  1223
# global_step:  1223
# learning rate:  0.0008942756
# train step:  1224
# global_step:  1224
# learning rate:  0.0008941939
# train step:  1225
# global_step:  1225
# learning rate:  0.00089411216
# train step:  1226
# global_step:  1226
# learning rate:  0.0008940305
# train step:  1227
# global_step:  1227
# learning rate:  0.0008939488
# train step:  1228
# global_step:  1228
# learning rate:  0.0008938671
# train step:  1229
# global_step:  1229
# learning rate:  0.00089378544
# train step:  1230
# global_step:  1230
# learning rate:  0.0008937038
# train step:  1231
# global_step:  1231
# learning rate:  0.0008936222
# train step:  1232
# global_step:  1232
# learning rate:  0.0008935405
# train step:  1233
# global_step:  1233
# learning rate:  0.0008934589
# train step:  1234
# global_step:  1234
# learning rate:  0.00089337723
# train step:  1235
# global_step:  1235
# learning rate:  0.0008932956
# train step:  1236
# global_step:  1236
# learning rate:  0.000893214
# train step:  1237
# global_step:  1237
# learning rate:  0.0008931324
# train step:  1238
# global_step:  1238
# learning rate:  0.0008930508
# train step:  1239
# global_step:  1239
# learning rate:  0.00089296926
# train step:  1240
# global_step:  1240
# learning rate:  0.00089288765
# train step:  1241
# global_step:  1241
# learning rate:  0.00089280604
# train step:  1242
# global_step:  1242
# learning rate:  0.0008927245
# train step:  1243
# global_step:  1243
# learning rate:  0.0008926429
# train step:  1244
# global_step:  1244
# learning rate:  0.00089256134
# train step:  1245
# global_step:  1245
# learning rate:  0.00089247985
# train step:  1246
# global_step:  1246
# learning rate:  0.0008923983
# train step:  1247
# global_step:  1247
# learning rate:  0.00089231675
# train step:  1248
# global_step:  1248
# learning rate:  0.0008922352
# train step:  1249
# global_step:  1249
# learning rate:  0.0008921537
# train step:  1250
# global_step:  1250
# learning rate:  0.00089207216
# train step:  1251
# global_step:  1251
# learning rate:  0.0008919907
# train step:  1252
# global_step:  1252
# learning rate:  0.00089190924
# train step:  1253
# global_step:  1253
# learning rate:  0.00089182775
# train step:  1254
# global_step:  1254
# learning rate:  0.00089174625
# train step:  1255
# global_step:  1255
# learning rate:  0.00089166476
# train step:  1256
# global_step:  1256
# learning rate:  0.00089158333
# train step:  1257
# global_step:  1257
# learning rate:  0.00089150184
# train step:  1258
# global_step:  1258
# learning rate:  0.0008914204
# train step:  1259
# global_step:  1259
# learning rate:  0.0008913389
# train step:  1260
# global_step:  1260
# learning rate:  0.0008912575
# train step:  1261
# global_step:  1261
# learning rate:  0.0008911761
# train step:  1262
# global_step:  1262
# learning rate:  0.0008910947
# train step:  1263
# global_step:  1263
# learning rate:  0.00089101325
# train step:  1264
# global_step:  1264
# learning rate:  0.0008909319
# train step:  1265
# global_step:  1265
# learning rate:  0.00089085044
# train step:  1266
# global_step:  1266
# learning rate:  0.00089076906
# train step:  1267
# global_step:  1267
# learning rate:  0.0008906877
# train step:  1268
# global_step:  1268
# learning rate:  0.0008906063
# train step:  1269
# global_step:  1269
# learning rate:  0.00089052494
# train step:  1270
# global_step:  1270
# learning rate:  0.00089044357
# train step:  1271
# global_step:  1271
# learning rate:  0.00089036225
# train step:  1272
# global_step:  1272
# learning rate:  0.0008902809
# train step:  1273
# global_step:  1273
# learning rate:  0.0008901995
# train step:  1274
# global_step:  1274
# learning rate:  0.00089011824
# train step:  1275
# global_step:  1275
# learning rate:  0.00089003687
# train step:  1276
# global_step:  1276
# learning rate:  0.00088995555
# train step:  1277
# global_step:  1277
# learning rate:  0.00088987424
# train step:  1278
# global_step:  1278
# learning rate:  0.000889793
# train step:  1279
# global_step:  1279
# learning rate:  0.00088971166
# train step:  1280
# global_step:  1280
# learning rate:  0.00088963035
# train step:  1281
# global_step:  1281
# learning rate:  0.0008895491
# train step:  1282
# global_step:  1282
# learning rate:  0.00088946783
# train step:  1283
# global_step:  1283
# learning rate:  0.0008893866
# train step:  1284
# global_step:  1284
# learning rate:  0.0008893053
# train step:  1285
# global_step:  1285
# learning rate:  0.00088922406
# train step:  1286
# global_step:  1286
# learning rate:  0.0008891428
# train step:  1287
# global_step:  1287
# learning rate:  0.0008890616
# train step:  1288
# global_step:  1288
# learning rate:  0.00088898034
# train step:  1289
# global_step:  1289
# learning rate:  0.00088889914
# train step:  1290
# global_step:  1290
# learning rate:  0.00088881794
# train step:  1291
# global_step:  1291
# learning rate:  0.0008887367
# train step:  1292
# global_step:  1292
# learning rate:  0.00088865554
# train step:  1293
# global_step:  1293
# learning rate:  0.00088857434
# train step:  1294
# global_step:  1294
# learning rate:  0.00088849314
# train step:  1295
# global_step:  1295
# learning rate:  0.000888412
# train step:  1296
# global_step:  1296
# learning rate:  0.0008883308
# train step:  1297
# global_step:  1297
# learning rate:  0.00088824966
# train step:  1298
# global_step:  1298
# learning rate:  0.0008881685
# train step:  1299
# global_step:  1299
# learning rate:  0.0008880874
# train step:  1300
# global_step:  1300
# learning rate:  0.0008880062
# train step:  1301
# global_step:  1301
# learning rate:  0.0008879251
# train step:  1302
# global_step:  1302
# learning rate:  0.00088784395
# train step:  1303
# global_step:  1303
# learning rate:  0.0008877628
# train step:  1304
# global_step:  1304
# learning rate:  0.0008876818
# train step:  1305
# global_step:  1305
# learning rate:  0.00088760065
# train step:  1306
# global_step:  1306
# learning rate:  0.0008875195
# train step:  1307
# global_step:  1307
# learning rate:  0.0008874385
# train step:  1308
# global_step:  1308
# learning rate:  0.0008873574
# train step:  1309
# global_step:  1309
# learning rate:  0.0008872763
# train step:  1310
# global_step:  1310
# learning rate:  0.0008871953
# train step:  1311
# global_step:  1311
# learning rate:  0.0008871142
# train step:  1312
# global_step:  1312
# learning rate:  0.0008870331
# train step:  1313
# global_step:  1313
# learning rate:  0.0008869521
# train step:  1314
# global_step:  1314
# learning rate:  0.0008868711
# train step:  1315
# global_step:  1315
# learning rate:  0.00088679005
# train step:  1316
# global_step:  1316
# learning rate:  0.000886709
# train step:  1317
# global_step:  1317
# learning rate:  0.000886628
# train step:  1318
# global_step:  1318
# learning rate:  0.00088654703
# train step:  1319
# global_step:  1319
# learning rate:  0.000886466
# train step:  1320
# global_step:  1320
# learning rate:  0.000886385
# train step:  1321
# global_step:  1321
# learning rate:  0.000886304
# train step:  1322
# global_step:  1322
# learning rate:  0.00088622305
# train step:  1323
# global_step:  1323
# learning rate:  0.00088614214
# train step:  1324
# global_step:  1324
# learning rate:  0.0008860611
# train step:  1325
# global_step:  1325
# learning rate:  0.00088598015
# train step:  1326
# global_step:  1326
# learning rate:  0.00088589924
# train step:  1327
# global_step:  1327
# learning rate:  0.00088581827
# train step:  1328
# global_step:  1328
# learning rate:  0.00088573736
# train step:  1329
# global_step:  1329
# learning rate:  0.00088565645
# train step:  1330
# global_step:  1330
# learning rate:  0.00088557554
# train step:  1331
# global_step:  1331
# learning rate:  0.00088549464
# train step:  1332
# global_step:  1332
# learning rate:  0.00088541367
# train step:  1333
# global_step:  1333
# learning rate:  0.0008853328
# train step:  1334
# global_step:  1334
# learning rate:  0.0008852519
# train step:  1335
# global_step:  1335
# learning rate:  0.00088517106
# train step:  1336
# global_step:  1336
# learning rate:  0.0008850902
# train step:  1337
# global_step:  1337
# learning rate:  0.00088500936
# train step:  1338
# global_step:  1338
# learning rate:  0.00088492845
# train step:  1339
# global_step:  1339
# learning rate:  0.0008848476
# train step:  1340
# global_step:  1340
# learning rate:  0.0008847668
# train step:  1341
# global_step:  1341
# learning rate:  0.0008846859
# train step:  1342
# global_step:  1342
# learning rate:  0.0008846051
# train step:  1343
# global_step:  1343
# learning rate:  0.00088452426
# train step:  1344
# global_step:  1344
# learning rate:  0.0008844435
# train step:  1345
# global_step:  1345
# learning rate:  0.0008843627
# train step:  1346
# global_step:  1346
# learning rate:  0.0008842819
# train step:  1347
# global_step:  1347
# learning rate:  0.0008842011
# train step:  1348
# global_step:  1348
# learning rate:  0.00088412035
# train step:  1349
# global_step:  1349
# learning rate:  0.0008840395
# train step:  1350
# global_step:  1350
# learning rate:  0.00088395877
# train step:  1351
# global_step:  1351
# learning rate:  0.000883878
# train step:  1352
# global_step:  1352
# learning rate:  0.00088379724
# train step:  1353
# global_step:  1353
# learning rate:  0.0008837165
# train step:  1354
# global_step:  1354
# learning rate:  0.0008836357
# train step:  1355
# global_step:  1355
# learning rate:  0.00088355504
# train step:  1356
# global_step:  1356
# learning rate:  0.0008834743
# train step:  1357
# global_step:  1357
# learning rate:  0.0008833936
# train step:  1358
# global_step:  1358
# learning rate:  0.00088331284
# train step:  1359
# global_step:  1359
# learning rate:  0.00088323216
# train step:  1360
# global_step:  1360
# learning rate:  0.0008831515
# train step:  1361
# global_step:  1361
# learning rate:  0.0008830708
# train step:  1362
# global_step:  1362
# learning rate:  0.0008829901
# train step:  1363
# global_step:  1363
# learning rate:  0.00088290946
# train step:  1364
# global_step:  1364
# learning rate:  0.00088282884
# train step:  1365
# global_step:  1365
# learning rate:  0.0008827481
# train step:  1366
# global_step:  1366
# learning rate:  0.0008826675
# train step:  1367
# global_step:  1367
# learning rate:  0.0008825868
# train step:  1368
# global_step:  1368
# learning rate:  0.00088250625
# train step:  1369
# global_step:  1369
# learning rate:  0.0008824256
# train step:  1370
# global_step:  1370
# learning rate:  0.00088234496
# train step:  1371
# global_step:  1371
# learning rate:  0.00088226434
# train step:  1372
# global_step:  1372
# learning rate:  0.0008821837
# train step:  1373
# global_step:  1373
# learning rate:  0.0008821031
# train step:  1374
# global_step:  1374
# learning rate:  0.00088202255
# train step:  1375
# global_step:  1375
# learning rate:  0.00088194193
# train step:  1376
# global_step:  1376
# learning rate:  0.0008818614
# train step:  1377
# global_step:  1377
# learning rate:  0.0008817808
# train step:  1378
# global_step:  1378
# learning rate:  0.00088170025
# train step:  1379
# global_step:  1379
# learning rate:  0.0008816197
# train step:  1380
# global_step:  1380
# learning rate:  0.00088153913
# train step:  1381
# global_step:  1381
# learning rate:  0.00088145863
# train step:  1382
# global_step:  1382
# learning rate:  0.0008813781
# train step:  1383
# global_step:  1383
# learning rate:  0.0008812976
# train step:  1384
# global_step:  1384
# learning rate:  0.00088121707
# train step:  1385
# global_step:  1385
# learning rate:  0.0008811365
# train step:  1386
# global_step:  1386
# learning rate:  0.000881056
# train step:  1387
# global_step:  1387
# learning rate:  0.0008809755
# train step:  1388
# global_step:  1388
# learning rate:  0.00088089507
# train step:  1389
# global_step:  1389
# learning rate:  0.0008808145
# train step:  1390
# global_step:  1390
# learning rate:  0.00088073406
# train step:  1391
# global_step:  1391
# learning rate:  0.0008806536
# train step:  1392
# global_step:  1392
# learning rate:  0.0008805731
# train step:  1393
# global_step:  1393
# learning rate:  0.00088049273
# train step:  1394
# global_step:  1394
# learning rate:  0.0008804123
# train step:  1395
# global_step:  1395
# learning rate:  0.00088033185
# train step:  1396
# global_step:  1396
# learning rate:  0.0008802514
# train step:  1397
# global_step:  1397
# learning rate:  0.00088017096
# train step:  1398
# global_step:  1398
# learning rate:  0.0008800906
# train step:  1399
# global_step:  1399
# learning rate:  0.0008800102
# train step:  1400
# global_step:  1400
# learning rate:  0.00087992975
# train step:  1401
# global_step:  1401
# learning rate:  0.00087984937
# train step:  1402
# global_step:  1402
# learning rate:  0.0008797689
# train step:  1403
# global_step:  1403
# learning rate:  0.0008796886
# train step:  1404
# global_step:  1404
# learning rate:  0.00087960827
# train step:  1405
# global_step:  1405
# learning rate:  0.0008795278
# train step:  1406
# global_step:  1406
# learning rate:  0.0008794475
# train step:  1407
# global_step:  1407
# learning rate:  0.0008793672
# train step:  1408
# global_step:  1408
# learning rate:  0.0008792868
# train step:  1409
# global_step:  1409
# learning rate:  0.0008792065
# train step:  1410
# global_step:  1410
# learning rate:  0.0008791262
# train step:  1411
# global_step:  1411
# learning rate:  0.0008790458
# train step:  1412
# global_step:  1412
# learning rate:  0.00087896554
# train step:  1413
# global_step:  1413
# learning rate:  0.0008788853
# train step:  1414
# global_step:  1414
# learning rate:  0.0008788049
# train step:  1415
# global_step:  1415
# learning rate:  0.0008787246
# train step:  1416
# global_step:  1416
# learning rate:  0.00087864435
# train step:  1417
# global_step:  1417
# learning rate:  0.0008785641
# train step:  1418
# global_step:  1418
# learning rate:  0.0008784838
# train step:  1419
# global_step:  1419
# learning rate:  0.0008784036
# train step:  1420
# global_step:  1420
# learning rate:  0.0008783233
# train step:  1421
# global_step:  1421
# learning rate:  0.00087824307
# train step:  1422
# global_step:  1422
# learning rate:  0.00087816286
# train step:  1423
# global_step:  1423
# learning rate:  0.0008780826
# train step:  1424
# global_step:  1424
# learning rate:  0.0008780024
# train step:  1425
# global_step:  1425
# learning rate:  0.00087792217
# train step:  1426
# global_step:  1426
# learning rate:  0.00087784196
# train step:  1427
# global_step:  1427
# learning rate:  0.00087776175
# train step:  1428
# global_step:  1428
# learning rate:  0.00087768154
# train step:  1429
# global_step:  1429
# learning rate:  0.0008776014
# train step:  1430
# global_step:  1430
# learning rate:  0.0008775212
# train step:  1431
# global_step:  1431
# learning rate:  0.00087744097
# train step:  1432
# global_step:  1432
# learning rate:  0.0008773609
# train step:  1433
# global_step:  1433
# learning rate:  0.0008772807
# train step:  1434
# global_step:  1434
# learning rate:  0.0008772005
# train step:  1435
# global_step:  1435
# learning rate:  0.00087712036
# train step:  1436
# global_step:  1436
# learning rate:  0.00087704026
# train step:  1437
# global_step:  1437
# learning rate:  0.0008769601
# train step:  1438
# global_step:  1438
# learning rate:  0.00087687996
# train step:  1439
# global_step:  1439
# learning rate:  0.00087679987
# train step:  1440
# global_step:  1440
# learning rate:  0.0008767198
# train step:  1441
# global_step:  1441
# learning rate:  0.0008766397
# train step:  1442
# global_step:  1442
# learning rate:  0.0008765596
# train step:  1443
# global_step:  1443
# learning rate:  0.0008764795
# train step:  1444
# global_step:  1444
# learning rate:  0.00087639946
# train step:  1445
# global_step:  1445
# learning rate:  0.0008763193
# train step:  1446
# global_step:  1446
# learning rate:  0.00087623927
# train step:  1447
# global_step:  1447
# learning rate:  0.00087615923
# train step:  1448
# global_step:  1448
# learning rate:  0.0008760792
# train step:  1449
# global_step:  1449
# learning rate:  0.00087599916
# train step:  1450
# global_step:  1450
# learning rate:  0.0008759191
# train step:  1451
# global_step:  1451
# learning rate:  0.0008758391
# train step:  1452
# global_step:  1452
# learning rate:  0.00087575905
# train step:  1453
# global_step:  1453
# learning rate:  0.0008756791
# train step:  1454
# global_step:  1454
# learning rate:  0.00087559904
# train step:  1455
# global_step:  1455
# learning rate:  0.00087551906
# train step:  1456
# global_step:  1456
# learning rate:  0.0008754391
# train step:  1457
# global_step:  1457
# learning rate:  0.0008753591
# train step:  1458
# global_step:  1458
# learning rate:  0.00087527913
# train step:  1459
# global_step:  1459
# learning rate:  0.0008751991
# train step:  1460
# global_step:  1460
# learning rate:  0.0008751192
# train step:  1461
# global_step:  1461
# learning rate:  0.0008750392
# train step:  1462
# global_step:  1462
# learning rate:  0.0008749593
# train step:  1463
# global_step:  1463
# learning rate:  0.00087487936
# train step:  1464
# global_step:  1464
# learning rate:  0.0008747994
# train step:  1465
# global_step:  1465
# learning rate:  0.00087471947
# train step:  1466
# global_step:  1466
# learning rate:  0.00087463955
# train step:  1467
# global_step:  1467
# learning rate:  0.0008745597
# train step:  1468
# global_step:  1468
# learning rate:  0.00087447977
# train step:  1469
# global_step:  1469
# learning rate:  0.0008743999
# train step:  1470
# global_step:  1470
# learning rate:  0.00087432
# train step:  1471
# global_step:  1471
# learning rate:  0.00087424007
# train step:  1472
# global_step:  1472
# learning rate:  0.0008741602
# train step:  1473
# global_step:  1473
# learning rate:  0.00087408035
# train step:  1474
# global_step:  1474
# learning rate:  0.00087400054
# train step:  1475
# global_step:  1475
# learning rate:  0.0008739207
# train step:  1476
# global_step:  1476
# learning rate:  0.00087384076
# train step:  1477
# global_step:  1477
# learning rate:  0.00087376096
# train step:  1478
# global_step:  1478
# learning rate:  0.00087368116
# train step:  1479
# global_step:  1479
# learning rate:  0.0008736013
# train step:  1480
# global_step:  1480
# learning rate:  0.0008735215
# train step:  1481
# global_step:  1481
# learning rate:  0.0008734417
# train step:  1482
# global_step:  1482
# learning rate:  0.00087336195
# train step:  1483
# global_step:  1483
# learning rate:  0.00087328214
# train step:  1484
# global_step:  1484
# learning rate:  0.0008732023
# train step:  1485
# global_step:  1485
# learning rate:  0.00087312254
# train step:  1486
# global_step:  1486
# learning rate:  0.0008730428
# train step:  1487
# global_step:  1487
# learning rate:  0.000872963
# train step:  1488
# global_step:  1488
# learning rate:  0.00087288325
# train step:  1489
# global_step:  1489
# learning rate:  0.0008728035
# train step:  1490
# global_step:  1490
# learning rate:  0.00087272376
# train step:  1491
# global_step:  1491
# learning rate:  0.0008726441
# train step:  1492
# global_step:  1492
# learning rate:  0.00087256427
# train step:  1493
# global_step:  1493
# learning rate:  0.0008724846
# train step:  1494
# global_step:  1494
# learning rate:  0.00087240484
# train step:  1495
# global_step:  1495
# learning rate:  0.00087232515
# train step:  1496
# global_step:  1496
# learning rate:  0.00087224547
# train step:  1497
# global_step:  1497
# learning rate:  0.0008721658
# train step:  1498
# global_step:  1498
# learning rate:  0.0008720861
# train step:  1499
# global_step:  1499
# learning rate:  0.0008720064
# train step:  1500
# global_step:  1500
# learning rate:  0.0008719268
# train step:  1501
# global_step:  1501
# learning rate:  0.0008718471
# train step:  1502
# global_step:  1502
# learning rate:  0.00087176746
# train step:  1503
# global_step:  1503
# learning rate:  0.00087168784
# train step:  1504
# global_step:  1504
# learning rate:  0.0008716082
# train step:  1505
# global_step:  1505
# learning rate:  0.0008715285
# train step:  1506
# global_step:  1506
# learning rate:  0.0008714489
# train step:  1507
# global_step:  1507
# learning rate:  0.00087136927
# train step:  1508
# global_step:  1508
# learning rate:  0.00087128964
# train step:  1509
# global_step:  1509
# learning rate:  0.00087121007
# train step:  1510
# global_step:  1510
# learning rate:  0.00087113044
# train step:  1511
# global_step:  1511
# learning rate:  0.00087105087
# train step:  1512
# global_step:  1512
# learning rate:  0.0008709713
# train step:  1513
# global_step:  1513
# learning rate:  0.00087089173
# train step:  1514
# global_step:  1514
# learning rate:  0.00087081216
# train step:  1515
# global_step:  1515
# learning rate:  0.0008707326
# train step:  1516
# global_step:  1516
# learning rate:  0.000870653
# train step:  1517
# global_step:  1517
# learning rate:  0.0008705735
# train step:  1518
# global_step:  1518
# learning rate:  0.000870494
# train step:  1519
# global_step:  1519
# learning rate:  0.0008704144
# train step:  1520
# global_step:  1520
# learning rate:  0.0008703349
# train step:  1521
# global_step:  1521
# learning rate:  0.0008702554
# train step:  1522
# global_step:  1522
# learning rate:  0.0008701759
# train step:  1523
# global_step:  1523
# learning rate:  0.0008700964
# train step:  1524
# global_step:  1524
# learning rate:  0.0008700169
# train step:  1525
# global_step:  1525
# learning rate:  0.0008699374
# train step:  1526
# global_step:  1526
# learning rate:  0.00086985796
# train step:  1527
# global_step:  1527
# learning rate:  0.00086977845
# train step:  1528
# global_step:  1528
# learning rate:  0.000869699
# train step:  1529
# global_step:  1529
# learning rate:  0.00086961954
# train step:  1530
# global_step:  1530
# learning rate:  0.0008695401
# train step:  1531
# global_step:  1531
# learning rate:  0.00086946064
# train step:  1532
# global_step:  1532
# learning rate:  0.0008693812
# train step:  1533
# global_step:  1533
# learning rate:  0.0008693018
# train step:  1534
# global_step:  1534
# learning rate:  0.00086922233
# train step:  1535
# global_step:  1535
# learning rate:  0.00086914294
# train step:  1536
# global_step:  1536
# learning rate:  0.00086906354
# train step:  1537
# global_step:  1537
# learning rate:  0.00086898415
# train step:  1538
# global_step:  1538
# learning rate:  0.00086890475
# train step:  1539
# global_step:  1539
# learning rate:  0.00086882536
# train step:  1540
# global_step:  1540
# learning rate:  0.00086874596
# train step:  1541
# global_step:  1541
# learning rate:  0.00086866657
# train step:  1542
# global_step:  1542
# learning rate:  0.00086858723
# train step:  1543
# global_step:  1543
# learning rate:  0.0008685079
# train step:  1544
# global_step:  1544
# learning rate:  0.0008684285
# train step:  1545
# global_step:  1545
# learning rate:  0.00086834916
# train step:  1546
# global_step:  1546
# learning rate:  0.0008682698
# train step:  1547
# global_step:  1547
# learning rate:  0.0008681905
# train step:  1548
# global_step:  1548
# learning rate:  0.0008681112
# train step:  1549
# global_step:  1549
# learning rate:  0.00086803193
# train step:  1550
# global_step:  1550
# learning rate:  0.0008679526
# train step:  1551
# global_step:  1551
# learning rate:  0.0008678733
# train step:  1552
# global_step:  1552
# learning rate:  0.00086779404
# train step:  1553
# global_step:  1553
# learning rate:  0.0008677147
# train step:  1554
# global_step:  1554
# learning rate:  0.0008676355
# train step:  1555
# global_step:  1555
# learning rate:  0.0008675562
# train step:  1556
# global_step:  1556
# learning rate:  0.0008674769
# train step:  1557
# global_step:  1557
# learning rate:  0.00086739764
# train step:  1558
# global_step:  1558
# learning rate:  0.0008673184
# train step:  1559
# global_step:  1559
# learning rate:  0.00086723914
# train step:  1560
# global_step:  1560
# learning rate:  0.0008671599
# train step:  1561
# global_step:  1561
# learning rate:  0.00086708076
# train step:  1562
# global_step:  1562
# learning rate:  0.00086700154
# train step:  1563
# global_step:  1563
# learning rate:  0.0008669223
# train step:  1564
# global_step:  1564
# learning rate:  0.0008668431
# train step:  1565
# global_step:  1565
# learning rate:  0.0008667639
# train step:  1566
# global_step:  1566
# learning rate:  0.0008666847
# train step:  1567
# global_step:  1567
# learning rate:  0.0008666055
# train step:  1568
# global_step:  1568
# learning rate:  0.00086652633
# train step:  1569
# global_step:  1569
# learning rate:  0.0008664471
# train step:  1570
# global_step:  1570
# learning rate:  0.000866368
# train step:  1571
# global_step:  1571
# learning rate:  0.00086628884
# train step:  1572
# global_step:  1572
# learning rate:  0.00086620974
# train step:  1573
# global_step:  1573
# learning rate:  0.0008661306
# train step:  1574
# global_step:  1574
# learning rate:  0.0008660514
# train step:  1575
# global_step:  1575
# learning rate:  0.0008659723
# train step:  1576
# global_step:  1576
# learning rate:  0.00086589315
# train step:  1577
# global_step:  1577
# learning rate:  0.00086581404
# train step:  1578
# global_step:  1578
# learning rate:  0.000865735
# train step:  1579
# global_step:  1579
# learning rate:  0.0008656559
# train step:  1580
# global_step:  1580
# learning rate:  0.0008655768
# train step:  1581
# global_step:  1581
# learning rate:  0.0008654977
# train step:  1582
# global_step:  1582
# learning rate:  0.00086541864
# train step:  1583
# global_step:  1583
# learning rate:  0.00086533953
# train step:  1584
# global_step:  1584
# learning rate:  0.00086526055
# train step:  1585
# global_step:  1585
# learning rate:  0.00086518144
# train step:  1586
# global_step:  1586
# learning rate:  0.0008651024
# train step:  1587
# global_step:  1587
# learning rate:  0.00086502335
# train step:  1588
# global_step:  1588
# learning rate:  0.0008649443
# train step:  1589
# global_step:  1589
# learning rate:  0.0008648653
# train step:  1590
# global_step:  1590
# learning rate:  0.0008647863
# train step:  1591
# global_step:  1591
# learning rate:  0.0008647073
# train step:  1592
# global_step:  1592
# learning rate:  0.0008646283
# train step:  1593
# global_step:  1593
# learning rate:  0.0008645493
# train step:  1594
# global_step:  1594
# learning rate:  0.0008644703
# train step:  1595
# global_step:  1595
# learning rate:  0.00086439133
# train step:  1596
# global_step:  1596
# learning rate:  0.00086431234
# train step:  1597
# global_step:  1597
# learning rate:  0.0008642334
# train step:  1598
# global_step:  1598
# learning rate:  0.0008641544
# train step:  1599
# global_step:  1599
# learning rate:  0.00086407544
# train step:  1600
# global_step:  1600
# learning rate:  0.00086399657
# train step:  1601
# global_step:  1601
# learning rate:  0.00086391764
# train step:  1602
# global_step:  1602
# learning rate:  0.00086383865
# train step:  1603
# global_step:  1603
# learning rate:  0.0008637597
# train step:  1604
# global_step:  1604
# learning rate:  0.00086368085
# train step:  1605
# global_step:  1605
# learning rate:  0.0008636019
# train step:  1606
# global_step:  1606
# learning rate:  0.00086352305
# train step:  1607
# global_step:  1607
# learning rate:  0.0008634441
# train step:  1608
# global_step:  1608
# learning rate:  0.0008633652
# train step:  1609
# global_step:  1609
# learning rate:  0.0008632864
# train step:  1610
# global_step:  1610
# learning rate:  0.0008632075
# train step:  1611
# global_step:  1611
# learning rate:  0.00086312863
# train step:  1612
# global_step:  1612
# learning rate:  0.00086304976
# train step:  1613
# global_step:  1613
# learning rate:  0.00086297095
# train step:  1614
# global_step:  1614
# learning rate:  0.0008628921
# train step:  1615
# global_step:  1615
# learning rate:  0.0008628132
# train step:  1616
# global_step:  1616
# learning rate:  0.0008627344
# train step:  1617
# global_step:  1617
# learning rate:  0.00086265564
# train step:  1618
# global_step:  1618
# learning rate:  0.00086257677
# train step:  1619
# global_step:  1619
# learning rate:  0.00086249795
# train step:  1620
# global_step:  1620
# learning rate:  0.0008624192
# train step:  1621
# global_step:  1621
# learning rate:  0.0008623404
# train step:  1622
# global_step:  1622
# learning rate:  0.0008622616
# train step:  1623
# global_step:  1623
# learning rate:  0.0008621828
# train step:  1624
# global_step:  1624
# learning rate:  0.00086210406
# train step:  1625
# global_step:  1625
# learning rate:  0.0008620253
# train step:  1626
# global_step:  1626
# learning rate:  0.0008619465
# train step:  1627
# global_step:  1627
# learning rate:  0.00086186774
# train step:  1628
# global_step:  1628
# learning rate:  0.00086178904
# train step:  1629
# global_step:  1629
# learning rate:  0.0008617103
# train step:  1630
# global_step:  1630
# learning rate:  0.00086163153
# train step:  1631
# global_step:  1631
# learning rate:  0.0008615529
# train step:  1632
# global_step:  1632
# learning rate:  0.00086147414
# train step:  1633
# global_step:  1633
# learning rate:  0.00086139544
# train step:  1634
# global_step:  1634
# learning rate:  0.00086131674
# train step:  1635
# global_step:  1635
# learning rate:  0.00086123805
# train step:  1636
# global_step:  1636
# learning rate:  0.00086115935
# train step:  1637
# global_step:  1637
# learning rate:  0.0008610807
# train step:  1638
# global_step:  1638
# learning rate:  0.000861002
# train step:  1639
# global_step:  1639
# learning rate:  0.0008609233
# train step:  1640
# global_step:  1640
# learning rate:  0.00086084474
# train step:  1641
# global_step:  1641
# learning rate:  0.00086076604
# train step:  1642
# global_step:  1642
# learning rate:  0.0008606874
# train step:  1643
# global_step:  1643
# learning rate:  0.00086060876
# train step:  1644
# global_step:  1644
# learning rate:  0.0008605301
# train step:  1645
# global_step:  1645
# learning rate:  0.0008604515
# train step:  1646
# global_step:  1646
# learning rate:  0.0008603729
# train step:  1647
# global_step:  1647
# learning rate:  0.0008602943
# train step:  1648
# global_step:  1648
# learning rate:  0.0008602157
# train step:  1649
# global_step:  1649
# learning rate:  0.0008601371
# train step:  1650
# global_step:  1650
# learning rate:  0.0008600585
# train step:  1651
# global_step:  1651
# learning rate:  0.00085997995
# train step:  1652
# global_step:  1652
# learning rate:  0.00085990137
# train step:  1653
# global_step:  1653
# learning rate:  0.0008598228
# train step:  1654
# global_step:  1654
# learning rate:  0.0008597442
# train step:  1655
# global_step:  1655
# learning rate:  0.0008596657
# train step:  1656
# global_step:  1656
# learning rate:  0.00085958716
# train step:  1657
# global_step:  1657
# learning rate:  0.00085950864
# train step:  1658
# global_step:  1658
# learning rate:  0.0008594301
# train step:  1659
# global_step:  1659
# learning rate:  0.00085935154
# train step:  1660
# global_step:  1660
# learning rate:  0.0008592731
# train step:  1661
# global_step:  1661
# learning rate:  0.00085919455
# train step:  1662
# global_step:  1662
# learning rate:  0.0008591161
# train step:  1663
# global_step:  1663
# learning rate:  0.00085903757
# train step:  1664
# global_step:  1664
# learning rate:  0.0008589591
# train step:  1665
# global_step:  1665
# learning rate:  0.00085888064
# train step:  1666
# global_step:  1666
# learning rate:  0.0008588021
# train step:  1667
# global_step:  1667
# learning rate:  0.0008587237
# train step:  1668
# global_step:  1668
# learning rate:  0.00085864525
# train step:  1669
# global_step:  1669
# learning rate:  0.0008585668
# train step:  1670
# global_step:  1670
# learning rate:  0.0008584884
# train step:  1671
# global_step:  1671
# learning rate:  0.0008584099
# train step:  1672
# global_step:  1672
# learning rate:  0.0008583315
# train step:  1673
# global_step:  1673
# learning rate:  0.00085825304
# train step:  1674
# global_step:  1674
# learning rate:  0.00085817464
# train step:  1675
# global_step:  1675
# learning rate:  0.00085809623
# train step:  1676
# global_step:  1676
# learning rate:  0.0008580178
# train step:  1677
# global_step:  1677
# learning rate:  0.0008579395
# train step:  1678
# global_step:  1678
# learning rate:  0.0008578611
# train step:  1679
# global_step:  1679
# learning rate:  0.0008577827
# train step:  1680
# global_step:  1680
# learning rate:  0.0008577043
# train step:  1681
# global_step:  1681
# learning rate:  0.0008576259
# train step:  1682
# global_step:  1682
# learning rate:  0.0008575476
# train step:  1683
# global_step:  1683
# learning rate:  0.0008574692
# train step:  1684
# global_step:  1684
# learning rate:  0.0008573909
# train step:  1685
# global_step:  1685
# learning rate:  0.0008573126
# train step:  1686
# global_step:  1686
# learning rate:  0.0008572343
# train step:  1687
# global_step:  1687
# learning rate:  0.00085715594
# train step:  1688
# global_step:  1688
# learning rate:  0.00085707766
# train step:  1689
# global_step:  1689
# learning rate:  0.0008569993
# train step:  1690
# global_step:  1690
# learning rate:  0.0008569211
# train step:  1691
# global_step:  1691
# learning rate:  0.00085684273
# train step:  1692
# global_step:  1692
# learning rate:  0.0008567645
# train step:  1693
# global_step:  1693
# learning rate:  0.00085668615
# train step:  1694
# global_step:  1694
# learning rate:  0.0008566079
# train step:  1695
# global_step:  1695
# learning rate:  0.00085652963
# train step:  1696
# global_step:  1696
# learning rate:  0.0008564514
# train step:  1697
# global_step:  1697
# learning rate:  0.00085637317
# train step:  1698
# global_step:  1698
# learning rate:  0.00085629494
# train step:  1699
# global_step:  1699
# learning rate:  0.00085621665
# train step:  1700
# global_step:  1700
# learning rate:  0.0008561385
# train step:  1701
# global_step:  1701
# learning rate:  0.00085606024
# train step:  1702
# global_step:  1702
# learning rate:  0.00085598207
# train step:  1703
# global_step:  1703
# learning rate:  0.0008559038
# train step:  1704
# global_step:  1704
# learning rate:  0.00085582567
# train step:  1705
# global_step:  1705
# learning rate:  0.00085574744
# train step:  1706
# global_step:  1706
# learning rate:  0.00085566926
# train step:  1707
# global_step:  1707
# learning rate:  0.00085559103
# train step:  1708
# global_step:  1708
# learning rate:  0.0008555129
# train step:  1709
# global_step:  1709
# learning rate:  0.00085543474
# train step:  1710
# global_step:  1710
# learning rate:  0.0008553566
# train step:  1711
# global_step:  1711
# learning rate:  0.0008552784
# train step:  1712
# global_step:  1712
# learning rate:  0.0008552003
# train step:  1713
# global_step:  1713
# learning rate:  0.00085512217
# train step:  1714
# global_step:  1714
# learning rate:  0.00085504405
# train step:  1715
# global_step:  1715
# learning rate:  0.0008549659
# train step:  1716
# global_step:  1716
# learning rate:  0.0008548878
# train step:  1717
# global_step:  1717
# learning rate:  0.0008548097
# train step:  1718
# global_step:  1718
# learning rate:  0.0008547316
# train step:  1719
# global_step:  1719
# learning rate:  0.00085465354
# train step:  1720
# global_step:  1720
# learning rate:  0.0008545754
# train step:  1721
# global_step:  1721
# learning rate:  0.00085449737
# train step:  1722
# global_step:  1722
# learning rate:  0.00085441925
# train step:  1723
# global_step:  1723
# learning rate:  0.00085434126
# train step:  1724
# global_step:  1724
# learning rate:  0.00085426314
# train step:  1725
# global_step:  1725
# learning rate:  0.00085418514
# train step:  1726
# global_step:  1726
# learning rate:  0.0008541071
# train step:  1727
# global_step:  1727
# learning rate:  0.00085402903
# train step:  1728
# global_step:  1728
# learning rate:  0.00085395103
# train step:  1729
# global_step:  1729
# learning rate:  0.000853873
# train step:  1730
# global_step:  1730
# learning rate:  0.000853795
# train step:  1731
# global_step:  1731
# learning rate:  0.00085371704
# train step:  1732
# global_step:  1732
# learning rate:  0.000853639
# train step:  1733
# global_step:  1733
# learning rate:  0.00085356104
# train step:  1734
# global_step:  1734
# learning rate:  0.000853483
# train step:  1735
# global_step:  1735
# learning rate:  0.00085340504
# train step:  1736
# global_step:  1736
# learning rate:  0.0008533271
# train step:  1737
# global_step:  1737
# learning rate:  0.0008532491
# train step:  1738
# global_step:  1738
# learning rate:  0.00085317116
# train step:  1739
# global_step:  1739
# learning rate:  0.00085309317
# train step:  1740
# global_step:  1740
# learning rate:  0.0008530153
# train step:  1741
# global_step:  1741
# learning rate:  0.00085293734
# train step:  1742
# global_step:  1742
# learning rate:  0.0008528594
# train step:  1743
# global_step:  1743
# learning rate:  0.00085278146
# train step:  1744
# global_step:  1744
# learning rate:  0.0008527035
# train step:  1745
# global_step:  1745
# learning rate:  0.00085262564
# train step:  1746
# global_step:  1746
# learning rate:  0.00085254776
# train step:  1747
# global_step:  1747
# learning rate:  0.0008524699
# train step:  1748
# global_step:  1748
# learning rate:  0.000852392
# train step:  1749
# global_step:  1749
# learning rate:  0.0008523141
# train step:  1750
# global_step:  1750
# learning rate:  0.00085223623
# train step:  1751
# global_step:  1751
# learning rate:  0.0008521584
# train step:  1752
# global_step:  1752
# learning rate:  0.0008520805
# train step:  1753
# global_step:  1753
# learning rate:  0.00085200265
# train step:  1754
# global_step:  1754
# learning rate:  0.0008519248
# train step:  1755
# global_step:  1755
# learning rate:  0.00085184706
# train step:  1756
# global_step:  1756
# learning rate:  0.0008517692
# train step:  1757
# global_step:  1757
# learning rate:  0.00085169135
# train step:  1758
# global_step:  1758
# learning rate:  0.0008516136
# train step:  1759
# global_step:  1759
# learning rate:  0.0008515357
# train step:  1760
# global_step:  1760
# learning rate:  0.00085145794
# train step:  1761
# global_step:  1761
# learning rate:  0.0008513802
# train step:  1762
# global_step:  1762
# learning rate:  0.00085130235
# train step:  1763
# global_step:  1763
# learning rate:  0.0008512246
# train step:  1764
# global_step:  1764
# learning rate:  0.0008511468
# train step:  1765
# global_step:  1765
# learning rate:  0.00085106905
# train step:  1766
# global_step:  1766
# learning rate:  0.0008509913
# train step:  1767
# global_step:  1767
# learning rate:  0.0008509136
# train step:  1768
# global_step:  1768
# learning rate:  0.0008508358
# train step:  1769
# global_step:  1769
# learning rate:  0.00085075805
# train step:  1770
# global_step:  1770
# learning rate:  0.00085068034
# train step:  1771
# global_step:  1771
# learning rate:  0.00085060264
# train step:  1772
# global_step:  1772
# learning rate:  0.0008505249
# train step:  1773
# global_step:  1773
# learning rate:  0.0008504472
# train step:  1774
# global_step:  1774
# learning rate:  0.0008503695
# train step:  1775
# global_step:  1775
# learning rate:  0.0008502918
# train step:  1776
# global_step:  1776
# learning rate:  0.00085021416
# train step:  1777
# global_step:  1777
# learning rate:  0.00085013645
# train step:  1778
# global_step:  1778
# learning rate:  0.0008500588
# train step:  1779
# global_step:  1779
# learning rate:  0.0008499811
# train step:  1780
# global_step:  1780
# learning rate:  0.00084990344
# train step:  1781
# global_step:  1781
# learning rate:  0.00084982585
# train step:  1782
# global_step:  1782
# learning rate:  0.00084974815
# train step:  1783
# global_step:  1783
# learning rate:  0.0008496705
# train step:  1784
# global_step:  1784
# learning rate:  0.0008495929
# train step:  1785
# global_step:  1785
# learning rate:  0.0008495153
# train step:  1786
# global_step:  1786
# learning rate:  0.00084943767
# train step:  1787
# global_step:  1787
# learning rate:  0.0008493601
# train step:  1788
# global_step:  1788
# learning rate:  0.0008492825
# train step:  1789
# global_step:  1789
# learning rate:  0.0008492049
# train step:  1790
# global_step:  1790
# learning rate:  0.00084912725
# train step:  1791
# global_step:  1791
# learning rate:  0.0008490497
# train step:  1792
# global_step:  1792
# learning rate:  0.0008489721
# train step:  1793
# global_step:  1793
# learning rate:  0.0008488946
# train step:  1794
# global_step:  1794
# learning rate:  0.00084881706
# train step:  1795
# global_step:  1795
# learning rate:  0.00084873947
# train step:  1796
# global_step:  1796
# learning rate:  0.00084866193
# train step:  1797
# global_step:  1797
# learning rate:  0.0008485844
# train step:  1798
# global_step:  1798
# learning rate:  0.00084850687
# train step:  1799
# global_step:  1799
# learning rate:  0.00084842934
# train step:  1800
# global_step:  1800
# learning rate:  0.0008483518
# train step:  1801
# global_step:  1801
# learning rate:  0.0008482743
# train step:  1802
# global_step:  1802
# learning rate:  0.00084819685
# train step:  1803
# global_step:  1803
# learning rate:  0.0008481193
# train step:  1804
# global_step:  1804
# learning rate:  0.00084804185
# train step:  1805
# global_step:  1805
# learning rate:  0.0008479644
# train step:  1806
# global_step:  1806
# learning rate:  0.0008478869
# train step:  1807
# global_step:  1807
# learning rate:  0.0008478094
# train step:  1808
# global_step:  1808
# learning rate:  0.000847732
# train step:  1809
# global_step:  1809
# learning rate:  0.0008476545
# train step:  1810
# global_step:  1810
# learning rate:  0.00084757706
# train step:  1811
# global_step:  1811
# learning rate:  0.00084749964
# train step:  1812
# global_step:  1812
# learning rate:  0.0008474222
# train step:  1813
# global_step:  1813
# learning rate:  0.0008473448
# train step:  1814
# global_step:  1814
# learning rate:  0.0008472674
# train step:  1815
# global_step:  1815
# learning rate:  0.0008471899
# train step:  1816
# global_step:  1816
# learning rate:  0.00084711256
# train step:  1817
# global_step:  1817
# learning rate:  0.00084703515
# train step:  1818
# global_step:  1818
# learning rate:  0.0008469578
# train step:  1819
# global_step:  1819
# learning rate:  0.00084688043
# train step:  1820
# global_step:  1820
# learning rate:  0.0008468031
# train step:  1821
# global_step:  1821
# learning rate:  0.0008467256
# train step:  1822
# global_step:  1822
# learning rate:  0.0008466483
# train step:  1823
# global_step:  1823
# learning rate:  0.00084657094
# train step:  1824
# global_step:  1824
# learning rate:  0.0008464936
# train step:  1825
# global_step:  1825
# learning rate:  0.0008464163
# train step:  1826
# global_step:  1826
# learning rate:  0.0008463389
# train step:  1827
# global_step:  1827
# learning rate:  0.0008462616
# train step:  1828
# global_step:  1828
# learning rate:  0.00084618427
# train step:  1829
# global_step:  1829
# learning rate:  0.00084610697
# train step:  1830
# global_step:  1830
# learning rate:  0.00084602967
# train step:  1831
# global_step:  1831
# learning rate:  0.00084595237
# train step:  1832
# global_step:  1832
# learning rate:  0.0008458751
# train step:  1833
# global_step:  1833
# learning rate:  0.0008457978
# train step:  1834
# global_step:  1834
# learning rate:  0.0008457206
# train step:  1835
# global_step:  1835
# learning rate:  0.0008456433
# train step:  1836
# global_step:  1836
# learning rate:  0.00084556604
# train step:  1837
# global_step:  1837
# learning rate:  0.00084548874
# train step:  1838
# global_step:  1838
# learning rate:  0.0008454115
# train step:  1839
# global_step:  1839
# learning rate:  0.00084533426
# train step:  1840
# global_step:  1840
# learning rate:  0.000845257
# train step:  1841
# global_step:  1841
# learning rate:  0.00084517983
# train step:  1842
# global_step:  1842
# learning rate:  0.0008451026
# train step:  1843
# global_step:  1843
# learning rate:  0.0008450254
# train step:  1844
# global_step:  1844
# learning rate:  0.0008449482
# train step:  1845
# global_step:  1845
# learning rate:  0.000844871
# train step:  1846
# global_step:  1846
# learning rate:  0.0008447938
# train step:  1847
# global_step:  1847
# learning rate:  0.0008447167
# train step:  1848
# global_step:  1848
# learning rate:  0.00084463943
# train step:  1849
# global_step:  1849
# learning rate:  0.00084456225
# train step:  1850
# global_step:  1850
# learning rate:  0.0008444851
# train step:  1851
# global_step:  1851
# learning rate:  0.00084440794
# train step:  1852
# global_step:  1852
# learning rate:  0.0008443308
# train step:  1853
# global_step:  1853
# learning rate:  0.0008442537
# train step:  1854
# global_step:  1854
# learning rate:  0.00084417657
# train step:  1855
# global_step:  1855
# learning rate:  0.00084409944
# train step:  1856
# global_step:  1856
# learning rate:  0.0008440223
# train step:  1857
# global_step:  1857
# learning rate:  0.0008439452
# train step:  1858
# global_step:  1858
# learning rate:  0.00084386807
# train step:  1859
# global_step:  1859
# learning rate:  0.000843791
# train step:  1860
# global_step:  1860
# learning rate:  0.00084371393
# train step:  1861
# global_step:  1861
# learning rate:  0.0008436368
# train step:  1862
# global_step:  1862
# learning rate:  0.00084355974
# train step:  1863
# global_step:  1863
# learning rate:  0.0008434827
# train step:  1864
# global_step:  1864
# learning rate:  0.0008434056
# train step:  1865
# global_step:  1865
# learning rate:  0.00084332854
# train step:  1866
# global_step:  1866
# learning rate:  0.00084325153
# train step:  1867
# global_step:  1867
# learning rate:  0.00084317446
# train step:  1868
# global_step:  1868
# learning rate:  0.00084309746
# train step:  1869
# global_step:  1869
# learning rate:  0.0008430204
# train step:  1870
# global_step:  1870
# learning rate:  0.0008429433
# train step:  1871
# global_step:  1871
# learning rate:  0.0008428663
# train step:  1872
# global_step:  1872
# learning rate:  0.0008427893
# train step:  1873
# global_step:  1873
# learning rate:  0.0008427123
# train step:  1874
# global_step:  1874
# learning rate:  0.00084263535
# train step:  1875
# global_step:  1875
# learning rate:  0.00084255834
# train step:  1876
# global_step:  1876
# learning rate:  0.0008424814
# train step:  1877
# global_step:  1877
# learning rate:  0.0008424044
# train step:  1878
# global_step:  1878
# learning rate:  0.0008423275
# train step:  1879
# global_step:  1879
# learning rate:  0.00084225053
# train step:  1880
# global_step:  1880
# learning rate:  0.0008421736
# train step:  1881
# global_step:  1881
# learning rate:  0.00084209663
# train step:  1882
# global_step:  1882
# learning rate:  0.0008420197
# train step:  1883
# global_step:  1883
# learning rate:  0.0008419428
# train step:  1884
# global_step:  1884
# learning rate:  0.00084186584
# train step:  1885
# global_step:  1885
# learning rate:  0.00084178895
# train step:  1886
# global_step:  1886
# learning rate:  0.000841712
# train step:  1887
# global_step:  1887
# learning rate:  0.0008416351
# train step:  1888
# global_step:  1888
# learning rate:  0.0008415582
# train step:  1889
# global_step:  1889
# learning rate:  0.0008414813
# train step:  1890
# global_step:  1890
# learning rate:  0.0008414044
# train step:  1891
# global_step:  1891
# learning rate:  0.0008413276
# train step:  1892
# global_step:  1892
# learning rate:  0.0008412507
# train step:  1893
# global_step:  1893
# learning rate:  0.00084117387
# train step:  1894
# global_step:  1894
# learning rate:  0.000841097
# train step:  1895
# global_step:  1895
# learning rate:  0.00084102014
# train step:  1896
# global_step:  1896
# learning rate:  0.0008409433
# train step:  1897
# global_step:  1897
# learning rate:  0.0008408665
# train step:  1898
# global_step:  1898
# learning rate:  0.00084078964
# train step:  1899
# global_step:  1899
# learning rate:  0.0008407128
# train step:  1900
# global_step:  1900
# learning rate:  0.00084063597
# train step:  1901
# global_step:  1901
# learning rate:  0.00084055925
# train step:  1902
# global_step:  1902
# learning rate:  0.0008404825
# train step:  1903
# global_step:  1903
# learning rate:  0.00084040564
# train step:  1904
# global_step:  1904
# learning rate:  0.00084032887
# train step:  1905
# global_step:  1905
# learning rate:  0.0008402521
# train step:  1906
# global_step:  1906
# learning rate:  0.0008401753
# train step:  1907
# global_step:  1907
# learning rate:  0.00084009854
# train step:  1908
# global_step:  1908
# learning rate:  0.0008400218
# train step:  1909
# global_step:  1909
# learning rate:  0.00083994505
# train step:  1910
# global_step:  1910
# learning rate:  0.0008398683
# train step:  1911
# global_step:  1911
# learning rate:  0.00083979155
# train step:  1912
# global_step:  1912
# learning rate:  0.0008397149
# train step:  1913
# global_step:  1913
# learning rate:  0.0008396382
# train step:  1914
# global_step:  1914
# learning rate:  0.00083956146
# train step:  1915
# global_step:  1915
# learning rate:  0.00083948474
# train step:  1916
# global_step:  1916
# learning rate:  0.000839408
# train step:  1917
# global_step:  1917
# learning rate:  0.0008393313
# train step:  1918
# global_step:  1918
# learning rate:  0.00083925464
# train step:  1919
# global_step:  1919
# learning rate:  0.000839178
# train step:  1920
# global_step:  1920
# learning rate:  0.00083910127
# train step:  1921
# global_step: D:\ProgramData\Anaconda3\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
#   from ._conv import register_converters as _register_converters
#  1921
# learning rate:  0.00083902467
# train step:  1922
# global_step:  1922
# learning rate:  0.000838948
# train step:  1923
# global_step:  1923
# learning rate:  0.00083887135
# train step:  1924
# global_step:  1924
# learning rate:  0.0008387947
# train step:  1925
# global_step:  1925
# learning rate:  0.0008387181
# train step:  1926
# global_step:  1926
# learning rate:  0.0008386414
# train step:  1927
# global_step:  1927
# learning rate:  0.00083856477
# train step:  1928
# global_step:  1928
# learning rate:  0.0008384882
# train step:  1929
# global_step:  1929
# learning rate:  0.0008384116
# train step:  1930
# global_step:  1930
# learning rate:  0.000838335
# train step:  1931
# global_step:  1931
# learning rate:  0.0008382584
# train step:  1932
# global_step:  1932
# learning rate:  0.0008381818
# train step:  1933
# global_step:  1933
# learning rate:  0.0008381052
# train step:  1934
# global_step:  1934
# learning rate:  0.00083802873
# train step:  1935
# global_step:  1935
# learning rate:  0.00083795213
# train step:  1936
# global_step:  1936
# learning rate:  0.00083787553
# train step:  1937
# global_step:  1937
# learning rate:  0.000837799
# train step:  1938
# global_step:  1938
# learning rate:  0.00083772244
# train step:  1939
# global_step:  1939
# learning rate:  0.0008376459
# train step:  1940
# global_step:  1940
# learning rate:  0.0008375694
# train step:  1941
# global_step:  1941
# learning rate:  0.0008374929
# train step:  1942
# global_step:  1942
# learning rate:  0.0008374164
# train step:  1943
# global_step:  1943
# learning rate:  0.00083733985
# train step:  1944
# global_step:  1944
# learning rate:  0.0008372633
# train step:  1945
# global_step:  1945
# learning rate:  0.0008371869
# train step:  1946
# global_step:  1946
# learning rate:  0.0008371104
# train step:  1947
# global_step:  1947
# learning rate:  0.00083703385
# train step:  1948
# global_step:  1948
# learning rate:  0.0008369574
# train step:  1949
# global_step:  1949
# learning rate:  0.00083688094
# train step:  1950
# global_step:  1950
# learning rate:  0.0008368045
# train step:  1951
# global_step:  1951
# learning rate:  0.000836728
# train step:  1952
# global_step:  1952
# learning rate:  0.0008366516
# train step:  1953
# global_step:  1953
# learning rate:  0.0008365751
# train step:  1954
# global_step:  1954
# learning rate:  0.00083649874
# train step:  1955
# global_step:  1955
# learning rate:  0.0008364223
# train step:  1956
# global_step:  1956
# learning rate:  0.0008363459
# train step:  1957
# global_step:  1957
# learning rate:  0.00083626946
# train step:  1958
# global_step:  1958
# learning rate:  0.00083619304
# train step:  1959
# global_step:  1959
# learning rate:  0.00083611667
# train step:  1960
# global_step:  1960
# learning rate:  0.0008360403
# train step:  1961
# global_step:  1961
# learning rate:  0.0008359639
# train step:  1962
# global_step:  1962
# learning rate:  0.0008358875
# train step:  1963
# global_step:  1963
# learning rate:  0.00083581114
# train step:  1964
# global_step:  1964
# learning rate:  0.00083573477
# train step:  1965
# global_step:  1965
# learning rate:  0.0008356584
# train step:  1966
# global_step:  1966
# learning rate:  0.0008355821
# train step:  1967
# global_step:  1967
# learning rate:  0.0008355058
# train step:  1968
# global_step:  1968
# learning rate:  0.0008354294
# train step:  1969
# global_step:  1969
# learning rate:  0.00083535304
# train step:  1970
# global_step:  1970
# learning rate:  0.0008352768
# train step:  1971
# global_step:  1971
# learning rate:  0.0008352005
# train step:  1972
# global_step:  1972
# learning rate:  0.0008351241
# train step:  1973
# global_step:  1973
# learning rate:  0.0008350478
# train step:  1974
# global_step:  1974
# learning rate:  0.0008349716
# train step:  1975
# global_step:  1975
# learning rate:  0.00083489524
# train step:  1976
# global_step:  1976
# learning rate:  0.000834819
# train step:  1977
# global_step:  1977
# learning rate:  0.0008347427
# train step:  1978
# global_step:  1978
# learning rate:  0.0008346665
# train step:  1979
# global_step:  1979
# learning rate:  0.0008345902
# train step:  1980
# global_step:  1980
# learning rate:  0.0008345139
# train step:  1981
# global_step:  1981
# learning rate:  0.0008344377
# train step:  1982
# global_step:  1982
# learning rate:  0.0008343615
# train step:  1983
# global_step:  1983
# learning rate:  0.0008342852
# train step:  1984
# global_step:  1984
# learning rate:  0.00083420903
# train step:  1985
# global_step:  1985
# learning rate:  0.00083413284
# train step:  1986
# global_step:  1986
# learning rate:  0.0008340566
# train step:  1987
# global_step:  1987
# learning rate:  0.0008339804
# train step:  1988
# global_step:  1988
# learning rate:  0.0008339042
# train step:  1989
# global_step:  1989
# learning rate:  0.000833828
# train step:  1990
# global_step:  1990
# learning rate:  0.0008337518
# train step:  1991
# global_step:  1991
# learning rate:  0.0008336757
# train step:  1992
# global_step:  1992
# learning rate:  0.0008335995
# train step:  1993
# global_step:  1993
# learning rate:  0.0008335233
# train step:  1994
# global_step:  1994
# learning rate:  0.00083344715
# train step:  1995
# global_step:  1995
# learning rate:  0.000833371
# train step:  1996
# global_step:  1996
# learning rate:  0.00083329494
# train step:  1997
# global_step:  1997
# learning rate:  0.0008332188
# train step:  1998
# global_step:  1998
# learning rate:  0.0008331426
# train step:  1999
# global_step:  1999
# learning rate:  0.00083306653

# [Done] exited with code=0 in 6215.654 seconds

