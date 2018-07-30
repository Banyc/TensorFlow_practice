import const
import tensorflow as tf 


class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps

        self.input_data = tf.placeholder(
            tf.int32,
            [batch_size,
            num_steps]
        )
        self.targets = tf.placeholder(
            tf.int32,
            [batch_size,
            num_steps]
        )

        dropout_keep_prob = const.LSTM_KEEP_PROB if is_training else 1.0
        lstm_cells = [
            tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.BasicLSTMCell(const.HIDDEN_SIZE),
                output_keep_prob=dropout_keep_prob
            )
            for _ in range(const.NUM_LAYERS)
        ]
        cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

        self.initial_state = cell.zero_state(batch_size, tf.float32)

        embedding = tf.get_variable(
            "embedding",
            [const.VOCAB_SIZE,
            const.HIDDEN_SIZE]
        )

        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        if is_training:
            inputs = tf.nn.dropout(inputs, const.EMBEDDING_KEEP_PROB)

        outputs = []
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: 
                    tf.get_variable_scope().reuse_variables()
                    cell_output, state = cell(inputs[:, time_step, :], state)
                    outputs.append(cell_output)

        output = tf.reshape(tf.concat(outputs, 1),
            [-1,
            const.HIDDEN_SIZE])

        if const.SHARE_EMB_AND_SOFTMAX:
            weight = tf.transpose(embedding)
        else:
            weight = tf.get_variable(
                "weight",
                [const.HIDDEN_SIZE,
                const.VOCAB_SIZE]
            )
            bias = tf.get_variable(
                "bias",
                [const.VOCAB_SIZE]
            )
            logits = tf.matmul(output, weight) + bias
        
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.targets, [-1]),
            logits=logits
        )
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state

        if not is_training: return

        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.cost, trainable_variables), const.MAX_GRAD_NORM
        )
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        self.train_op = optimizer.apply_gradients(
            list(zip(grads, trainable_variables))
        )
