# Sex Identification by Names

The files here are amended from http://blog.topspeedsnail.com/archives/10833

are reduced coupling (did not do enough)

## Type of Problem

classification in convolutional neural network plus vocabulary transformation

## Word proceeding

referred from
 - [Introduction of Embedding layer](https://blog.csdn.net/u013713117/article/details/55049808)
 - [Example of matrix for Embedding layer](https://towardsdatascience.com/deep-learning-4-embedding-layers-f9a02d55ac12)
 - [Process of "tf.nn.embedding_lookup()"](https://blog.csdn.net/laolu1573/article/details/77170407#!/_c183ucg7nhg)

~~1. text -> one hot vector~~
~~2. one hot vector -> embedding vector~~

Ingredients
- a list of vocabulary with respective indexes
- several sentences for process

Preprocess
1. treat the sentences as a set of data in a batch while each word in lines as a minimal element.
2. find out the relative index of each word in each line from the list of vocabulary, and record them in the form of list for example, [[3, 6, 5, 2, ...], ...]. In order to form a matrix, each row (the inner list) should share the same length of elements. To reach the requirement, the first index of the vocabulary list should be a blank, and each row should be extended by appending with 0 (if the sentence represented by the row does not consist of the largest amounts of words). The matrix here, identified as "input_ids", is NOT the final embedding layer. The shape is [num_of_sentences, words_count_of_the_sentence_with_largest_words_count], or [batch_size, max_sentence_len].
3. randomly generate a trainable matrix with [the length of the vocabulary list, embedding_size] in shape. "Embedding_size" here is not a fixed variable, which is a hyperparameter. Identify it as "weights". [To learn more](https://towardsdatascience.com/deep-learning-4-embedding-layers-f9a02d55ac12)
4. return a selected matrix through tf.nn.embedding_lookup(weights, input_ids). The matrix is a layer with [batch_size, max_len, embedding_size] in shape. [To learn more](https://blog.csdn.net/laolu1573/article/details/77170407#!/_c183ucg7nhg)
5. extend a dimension to the last matrix in order to be the input layer of convolutional neural network. The final embedding layer is in the shape of [batch_size, max_sentence_len, embedding_size, 1].
6. the later steps are illuminated [here](https://blog.csdn.net/u013713117/article/details/55049808), which is the like of CNN.
PS: read codes from model.py and predict.py to understand.


## Deployment

Run "train.py" file to train model.

Run "predict.py" file to see prediction of the newly feeded names.

