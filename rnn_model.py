import numpy as np
import tensorflow as tf
import util.reader as reader

DATA_PATH = "dataset/ptb"

HIDDEN_SIZE = 200
NUM_LAYERS = 2
VOCAB_SIZE = 10000

LEARNING_RATE = 1.0
TRAIN_BATCH_SIZE = 64
TRAIN_NUM_STEP = 20

EVAL_BATCH_SIZE = 1
EVAL_NUM_STEP = 1
NUM_EPOCH = 2
KEEP_PROB = 0.5
MAX_GRAD_NORM = 5

## time_step * batch * vector_dim_input
class PTBModel(object):

    def __init__(self, is_training, batch_size, num_steps):

        self.batch_size = batch_size
        self.num_steps = num_steps

        # 定义输入层。
        ## 重要的一点，是数据集里面，本身就是索引
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        # 定义使用LSTM结构及训练时使用dropout。
        ## 隐藏层n = 200 就是state的Wfn
        ## 这也是唯一一个参数,就是那个 N
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE)
        if is_training:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=KEEP_PROB)
        ## 纵向看
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * NUM_LAYERS)
        # 初始化最初的状态。batch个样本，hidden_size已经封装在cell构造里了
        self.initial_state = cell.zero_state(batch_size, tf.float32)
        ## voc * hidden hidden的向量表达这个词，
        ##
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])

        # 将原本单词ID转为单词向量。化成 time * batch * XN
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        if is_training:
            inputs = tf.nn.dropout(inputs, KEEP_PROB)


        # 定义输出列表。
        outputs = []
        state = self.initial_state
        with tf.variable_scope("LRN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                ## 每一个时刻的输出  , state重用
                ## 这里output的维度 是hidden_size，与s一样，也用了N的维度
                cell_output, state = cell(inputs[:, time_step, :], state)
                ## print(state)
                outputs.append(cell_output)
        ## 20step * 64batch *  200N
        #print(outputs)
        # ## 连成一个 ,1代表第二维连接 ，
        # 保留 hidden ，其他平展
        ## 平展的是 time_step * batch batch代表多个word一句话,step代表多句话
        ## 展开的意思是，每一个word的展开
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])
        ## 1280*200
        # print(output)
        ## 开始从一个index,用hidden_size向量描述这个index
        ## 现在有了这个向量的值,转化成为word_vector
        ## 复用对象
        weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable("bias", [VOCAB_SIZE])

        ##1280*10000
        logits = tf.matmul(output, weight) + bias
        #
        # 定义交叉熵损失函数和平均损失。
        ## 函数用法见test2.py
        ## 返回值是 1D batch-sized float Tensor: The log-perplexity for each sequence.
        ## 这个是误差计算，与target那个index
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            ## 1280 * 10000
            [logits],
            ## 1280 *1 注意，这个1不超过10000 1D tensor int
            [tf.reshape(self.targets, [-1])],
            ##权重 weigths
            [tf.ones([batch_size * num_steps], dtype=tf.float32)])
        ## loss是1D的，每一个样本的loss
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state

        # 只在训练模型时定义反向传播操作。
        if not is_training: return
        ## 需要训练的变量s
        trainable_variables = tf.trainable_variables()
        # 控制梯度大小，定义优化方法和训练步骤。
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))


## train_model
def run_epoch(session, model, data, train_op, output_log, epoch_size):
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    # 训练一个epoch。
    for step in range(epoch_size):
        x, y = session.run(data)
        ## 填充
        cost, state, _ = session.run([model.cost, model.final_state, train_op],
                                        {model.input_data: x, model.targets: y, model.initial_state: state})
        total_costs += cost
        iters += model.num_steps

        if output_log and step % 100 == 0:
            print("After %d steps, perplexity is %.3f" % (step, np.exp(total_costs / iters)))
    return np.exp(total_costs / iters)



def main():
    train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)

    # 计算一个epoch需要训练的次数
    train_data_len = len(train_data)
    train_batch_len = train_data_len // TRAIN_BATCH_SIZE
    train_epoch_size = (train_batch_len - 1) // TRAIN_NUM_STEP

    valid_data_len = len(valid_data)
    ## 输入一个向量
    valid_batch_len = valid_data_len // EVAL_BATCH_SIZE
    valid_epoch_size = (valid_batch_len - 1) // EVAL_NUM_STEP

    test_data_len = len(test_data)
    test_batch_len = test_data_len // EVAL_BATCH_SIZE
    test_epoch_size = (test_batch_len - 1) // EVAL_NUM_STEP

    initializer = tf.random_uniform_initializer(-0.05, 0.05)


    with tf.variable_scope("language_model", reuse=None, initializer=initializer):
        ## 两套模型 这一套is_training = true
        train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)
    ## 重用变量,对于那些get_variable有效
    ## 注意，new Model的时候，虽然传入的batch 和 step 不同，但是 w , embedding,b并没有变化
    with tf.variable_scope("language_model", reuse=True, initializer=initializer):
        eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)

    # 训练模型。
    with tf.Session() as session:
        tf.global_variables_initializer().run()

        train_queue = reader.ptb_producer(train_data, train_model.batch_size, train_model.num_steps)
        ## 评估
        eval_queue = reader.ptb_producer(valid_data, eval_model.batch_size, eval_model.num_steps)
        test_queue = reader.ptb_producer(test_data, eval_model.batch_size, eval_model.num_steps)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)

        for i in range(1):
            print("In iteration: %d" % (i + 1))
            run_epoch(session, train_model, train_queue, train_model.train_op, True, train_epoch_size)

            valid_perplexity = run_epoch(session, eval_model, eval_queue, tf.no_op(), False, valid_epoch_size)
            print("Epoch: %d Validation Perplexity: %.3f" % (i + 1, valid_perplexity))

        test_perplexity = run_epoch(session, eval_model, test_queue, tf.no_op(), False, test_epoch_size)
        print("Test Perplexity: %.3f" % test_perplexity)

        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    main()