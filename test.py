## 测试embedding

import numpy as np
import tensorflow as tf

HIDDEN_SIZE = 200
VOCAB_SIZE = 10000
BATCH_SIZE = 64
NUM_STEP = 35
# X = [1,2]
#
# # for i in range(len(X)):
# #     print(len(X))
# #     print(X[i])
#
#
# t1 = np.array([5, 6])
# t2 = np.array([[1, 2], [3, 4]])
# print(np.dot(t1,t2))
# print(np.dot(t2,t1))
input_data = tf.placeholder(tf.int32, [BATCH_SIZE, NUM_STEP])
print(input_data)

embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])
print(embedding)
## 从embedding 找到
inputs = tf.nn.embedding_lookup(embedding, input_data)
print(embedding)
print(inputs)