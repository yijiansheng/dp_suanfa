import tensorflow as tf
import pickle
import numpy as np
import ast
from collections import defaultdict

train_data = 'dataset/comprehen/train.vec'
valid_data = 'dataset/comprehen/valid.vec'

#
BATCH_SIZE = 64
TRAIN_FILE = open(train_data)
word2idx, content_length, question_length, VOCAB_SIZE = pickle.load(open('dataset/comprehen/vocab.data', "rb"))


def get_next_batch():
    X = []
    Q = []
    A = []
    for i in range(BATCH_SIZE):
        for line in TRAIN_FILE:
            ## 一行代表一篇文章的信息
            ## 一篇文章有 800Cont , 100Q,1 Answer
            line = ast.literal_eval(line.strip())
            X.append(line[0])
            Q.append(line[1])
            A.append(line[2][0])
            break

    if len(X) == BATCH_SIZE:
        return X, Q, A
    else:
        TRAIN_FILE.seek(0)
        return get_next_batch()


def get_test_batch():
    with open(valid_data) as f:
        X = []
        Q = []
        A = []
        ## 这个是全部数据放到一个list
        for line in f:
            line = ast.literal_eval(line.strip())
            X.append(line[0])
            Q.append(line[1])
            A.append(line[2][0])
        return X, Q, A

#x,q,a = get_next_batch()
# print("x",x)
# print("q",q)
# print("a",a)






samples = get_next_batch()
## content
## X输入是64个文章 , 每篇文章的长度都是 800
X = tf.placeholder(tf.int32, [BATCH_SIZE, content_length])
## question输入是64个
Q = tf.placeholder(tf.int32, [BATCH_SIZE, question_length])
## 64个 answer,直接是indexs
A = tf.placeholder(tf.int32, [BATCH_SIZE])


keep_prob = tf.placeholder(tf.float32)


def glimpse(weights, bias, encodings, inputs):
    ## 256 * 512
    weights = tf.nn.dropout(weights, keep_prob)
    ## 64 * 512
    inputs = tf.nn.dropout(inputs, keep_prob)
    #print("front",inputs)
    ## 64 * 256 64个样本，
    attention = tf.transpose(tf.matmul(weights, tf.transpose(inputs)) + bias)
  #  print("back",tf.transpose(inputs))
    print("attention",attention)
    print("attention-1",tf.expand_dims(attention, -1))
    attention = tf.matmul(encodings, tf.expand_dims(attention, -1))
    print("encodings",encodings)
    print("attention-result",attention)
    attention = tf.nn.softmax(tf.squeeze(attention, -1))
    return attention, tf.reduce_sum(tf.expand_dims(attention, -1) * encodings, 1)

## 向量放在input的最后一维上面
## 这个算法是AOA，它是基于doc层面的，每次进入batch个doc，不是batch个sentence
STATE_SIZE = 128
XM_SIZE = 200
def neural_attention():
    embeddings = tf.Variable(tf.random_normal([VOCAB_SIZE, XM_SIZE], stddev=0.22), dtype=tf.float32)
    tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), [embeddings])


    with tf.variable_scope('encode'):
        with tf.variable_scope('X'):
            X_lens = tf.reduce_sum(tf.sign(tf.abs(X)), 1)
            embedded_X = tf.nn.embedding_lookup(embeddings, X)
            encoded_X = tf.nn.dropout(embedded_X, keep_prob)
            gru_cell = tf.contrib.rnn.GRUCell(STATE_SIZE)
            ## 这里没有步长,只有batch * batch(一篇article的batch) * em
            ## 联想lstm的输入,batch * timestep * em 输入是batch行句子，每一行句子都是相同的timestep
            ## 但不要太执着于输入输出，只是形式
            ## 要看训练的数据 和 想要得到的结果
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(gru_cell, gru_cell, encoded_X, sequence_length=X_lens, dtype=tf.float32, swap_memory=True)
            ## 64, 804, 128 到 256 相当于一个word的index，有256个特征,这个跟xm没有关系
            encoded_X = tf.concat(outputs, 2)
            #print(outputs)
            ## 64, 804, 256
            ## print(encoded_X)

        with tf.variable_scope('Q'):
            Q_lens = tf.reduce_sum(tf.sign(tf.abs(Q)), 1)
            embedded_Q = tf.nn.embedding_lookup(embeddings, Q)
            ## 这个是输入
            encoded_Q = tf.nn.dropout(embedded_Q, keep_prob)
            ## print(encoded_Q)
            gru_cell = tf.contrib.rnn.GRUCell(STATE_SIZE)
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(gru_cell, gru_cell, encoded_Q,
                                                                     sequence_length=Q_lens, dtype=tf.float32,
                                                                     swap_memory=True)
            ## 64  116  128*2=256
            encoded_Q = tf.concat(outputs,2)
            ## print(encoded_Q)


        ## question 2->4
        W_q = tf.Variable(tf.random_normal([2 * STATE_SIZE, 4 * STATE_SIZE], stddev=0.22), dtype=tf.float32)
        b_q = tf.Variable(tf.random_normal([2 * STATE_SIZE, 1], stddev=0.22), dtype=tf.float32)

        ## document 2->6
        W_d = tf.Variable(tf.random_normal([2 * STATE_SIZE, 6 * STATE_SIZE], stddev=0.22), dtype=tf.float32)
        b_d = tf.Variable(tf.random_normal([2 * STATE_SIZE, 1], stddev=0.22), dtype=tf.float32)

        ## 10倍 state入，2倍出
        g_q = tf.Variable(tf.random_normal([10 * STATE_SIZE, 2 * STATE_SIZE], stddev=0.22), dtype=tf.float32)
        g_d = tf.Variable(tf.random_normal([10 * STATE_SIZE, 2 * STATE_SIZE], stddev=0.22), dtype=tf.float32)

        ##
        with tf.variable_scope('attend') as scope:

            infer_gru = tf.contrib.rnn.GRUCell(4 * STATE_SIZE)
            ## batch(64) * 4Sn
            infer_state = infer_gru.zero_state(BATCH_SIZE, tf.float32)
            for iter_step in range(8):
                if iter_step > 0:
                    scope.reuse_variables()

                _, q_glimpse = glimpse(W_q, b_q, encoded_Q, infer_state)

                d_attention, d_glimpse = glimpse(W_d, b_d, encoded_X, tf.concat([infer_state, q_glimpse], 1))
                gate_concat = tf.concat([infer_state,  q_glimpse,  d_glimpse,  q_glimpse * d_glimpse], 1)

                r_d = tf.sigmoid(tf.matmul(gate_concat, g_d))
                r_d = tf.nn.dropout(r_d, keep_prob)
                r_q = tf.sigmoid(tf.matmul(gate_concat, g_q))
                r_q = tf.nn.dropout(r_q, keep_prob)

                combined_gated_glimpse = tf.concat([r_q * q_glimpse, r_d * d_glimpse], 1)
                _, infer_state = infer_gru(combined_gated_glimpse, infer_state)

        return tf.to_float(tf.sign(tf.abs(X))) * d_attention



neural_attention()