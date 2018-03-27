## 看到每次输出

import numpy as np
import tensorflow as tf
import util.reader as reader

DATA_PATH = "dataset/ptb"

STATE_SIZE = 200 ## N=200
VOCAB_SIZE = 10000

# NUM_LAYERS = 3
LEARNING_RATE = 1.0 ## 学习率
TRAIN_BATCH_SIZE = 64 ## 一批样本
TRAIN_NUM_STEP = 20 ## 时间步
KEEP_PROB = 0.5
MAX_GRAD_NORM = 5

class FBModel(object):

    def __init__(self, is_training, batch_size, num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps
        ## 时间步就是多少个cell，所以两次的结构在这里不同
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])
        ## state的Wfn ,Sn * Wfn
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(STATE_SIZE)
        ## 纵向两层,一个层叠的cell，可以不使用
        #cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * NUM_LAYERS)
        cell = lstm_cell
        ## 一个X向量对应一个state,[batch_size x state_size]
        self.initial_state = cell.zero_state(batch_size, tf.float32)
        ## 分布表征 xm作为xn，用一样的维度
        ## 但是这个映射关系是可以改变的
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, STATE_SIZE])
        self.embedding = embedding
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        outputs = []
        state = self.initial_state
        with tf.variable_scope("LRNO"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                ## 调用了 __call__ 方法
                ## core_rnn_cell_impl.py
                ## 会根据input的维度，来匹配linear方法里面  xm 对应的 Uxm的数量
                ## 但是每一个Hout，肯定是N个输出。
                ## Most basic RNN: output = new_state = act(W * input + U * state + B).
                cell_output, state = cell(inputs[:, time_step, :], state)
                ## 20 * 64 * 200
                outputs.append(cell_output)

        ## 1280 * 200 200是Hout Hout
        output = tf.reshape(tf.concat(outputs, 1), [-1, STATE_SIZE])
        weight = tf.get_variable("weight", [STATE_SIZE, VOCAB_SIZE])
        bias = tf.get_variable("bias", [VOCAB_SIZE])
        ## output只是每一层的Hout,需要最后全连接
        logits = tf.matmul(output, weight) + bias

        ## 取一个final_state
        self.final_state = state
        ## 作为这次的预测值
        predict_vector = tf.argmax(logits, axis=1)
        predict_vector = tf.reshape(predict_vector, [batch_size, num_steps])
        self.predict_vector = predict_vector


        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            ## 1280 * 10000
            [logits],
            ## 1280 *1 tensor 1D 值为index
            [tf.reshape(self.targets, [-1])],
            ##权重 weigths 默认一样重要
            [tf.ones([batch_size * num_steps], dtype=tf.float32)]
        )
        ## 取一个平均值 ,最后剩下一个cost
        ## 定义出来要优化的值
        self.cost = tf.reduce_sum(loss) / batch_size


        ## 如果不是训练，则不必反向传播
        if not is_training: return
        trainable_variables = tf.trainable_variables()
        # for v in trainable_variables:
        #     print(v.name,v)
        # print(lstm_cell.state_size)
        # print(lstm_cell.output_size)

        ## 控制一个最大平方和
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)
        ## 还是那五个变量
        # for v in grads:
        #     print(v.name)
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        ## 表示执行一次GD
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

wordsDict = reader._build_vocab(DATA_PATH + "/ptb.train.txt")

## train_model 主要就是更新五个参数
def run_train(session, model, data, epoch_size):
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    #print("state0:",state)

    ## 一次进入一张纸
    ## 那么这一次训练，就会有一个总的cost和state，Model是给这张纸准备的
    for step in range(epoch_size):
        ## 记住，这里y比x领先一个time_step
        ## x = 64 * 20 有64行 64组 每一行代表20步
        ## 输入的时候64组一起进入，每一次都是一句的相同位置的元素
        ## cell每次处理一个batch的数据，经过time_step次
        x, y = session.run(data)
        ##print(x.shape)
        # state,embedding1 = session.run(
        #     [model.initial_state,model.embedding],
        #     {model.input_data: x, model.targets: y, model.initial_state: state})
        # #print("state1:",state)
        # print("embedding1:",embedding1)
        #
        # state,embedding2 = session.run(
        #     [model.final_state,model.embedding],
        #     {model.input_data: x, model.targets: y, model.initial_state: state})
        # #print("state2:",state)
        # print("em_差值2:",embedding2-embedding1)
        #
        # state, embedding3,_ = session.run(
        #     [model.final_state, model.embedding,model.train_op],
        #     {model.input_data: x, model.targets: y, model.initial_state: state})
        # #print("state3:", state)
        # print("em_差值3:", embedding3 - embedding1)
        #
        # state, embedding4, _ = session.run(
        #     [model.final_state, model.embedding, model.train_op],
        #     {model.input_data: x, model.targets: y, model.initial_state: state})
        # #print("state4:", state)
        # print("em_差值4:", embedding4 - embedding1)


        # state, embedding = session.run(
        #     [model.initial_state, model.embedding],
        #     {model.input_data: x, model.targets: y, model.initial_state: state})
        # print("state2:", state)

        ## 这里表示执行，一个batch的数据进去
        cost, state, _ ,predict_vector,embedding1 = session.run(
            [model.cost, model.final_state , model.train_op,model.predict_vector, model.embedding],
            {model.input_data: x, model.targets: y, model.initial_state: state})
    #    print("embedding1:",embedding1)
    #    print("state_t1:",state)

        # _, state, _, _, embedding2 = session.run(
        #     [model.cost, model.final_state, train_op, model.predict_vector, model.embedding],
        #     {model.input_data: x, model.targets: y, model.initial_state: state})

    #    print("embedding2:",embedding2)
    #    print("state_t2:",state)
        # yVector = []
        # for line in y:
        #     lVector = []
        #     for element in line:
        #         for k, v in wordsDict.items():
        #             if v == element:
        #                 lVector.append(k)
        #     yVector.append(lVector)
        # print(yVector)
        # preVector = []
        # for line in predict_vector:
        #     lVector = []
        #     for element in line:
        #         for k, v in wordsDict.items():
        #             if v == element:
        #                 lVector.append(k)
        #     preVector.append(lVector)
        # print(preVector)

        ## 观察变量的值

        print("本次损失值",cost)
        total_costs += cost
        iters += model.num_steps
        print("After %d steps, perplexity is %.3f" % (step, np.exp(total_costs / iters)))
##        print("embedding1:",embedding1)

    return np.exp(total_costs / iters)



def run_test(session,test_model,test_queue,test_epoch_size):
    state = session.run(test_model.initial_state)
    for step in range(test_epoch_size):
        x, y = session.run(test_queue)
        state, predict_vector, embedding = session.run(
            [test_model.final_state,test_model.predict_vector, test_model.embedding],
            {test_model.input_data: x,
             test_model.targets: y,
             test_model.initial_state: state})
        print("testmodel:",predict_vector)


def trainModel():
    model = FBModel(True,TRAIN_BATCH_SIZE,TRAIN_NUM_STEP)
    train_data, _, test_data, _ = reader.ptb_raw_data(DATA_PATH)

    # 计算一个epoch需要训练的次数
    ## 训练10轮
    train_epoch_size = 50
    ## 测试两轮
    test_epoch_size = 2
    ## 这个固定写法，scope一样的时候，不能都为true
    with tf.variable_scope("language_model"):
        ## 两套模型 这一套is_training = true
        train_model = FBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)
    with tf.variable_scope("language_model",reuse=True):
        test_model = FBModel(False, 2, 2)
    # 训练模型。
    with tf.Session() as session:
        ## 这个地方的run是必须加的，类似session.run(Init)
        tf.global_variables_initializer().run()

        ## 训练队列
        train_queue = reader.ptb_producer(train_data, train_model.batch_size, train_model.num_steps)
        ## 测试队列
        test_queue = reader.ptb_producer(test_data, test_model.batch_size, test_model.num_steps)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)

        ## 训练模型
        run_train(session, train_model, train_queue, train_epoch_size)

        ## 用模型进行测试
        run_test(session,test_model,test_queue,test_epoch_size)

        # x, y = session.run(train_queue)
        # print("finalembedding:",session.run(train_model.embedding,feed_dict={
        #     train_model.input_data: x, train_model.targets: y
        #
        # }))
##         print("testembedding:",session.run(test_model.embedding))

        coord.request_stop()
        coord.join(threads)

    return test_model,test_data


trainModel()
