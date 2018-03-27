import tensorflow as tf
## 测试交叉熵

A = tf.random_normal([5, 4], dtype=tf.float32)
B = tf.constant([1, 2, 1, 3, 3], dtype=tf.int32)
w = tf.ones([5], dtype=tf.float32)


D = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
    [A],
    [B],
    [w])

with tf.Session() as sess:
    print(sess.run(A))
    print(sess.run(B))
    print(sess.run(D))