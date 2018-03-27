import tensorflow as tf
import util.reader as reader

DATA_PATH = "dataset/ptb"
train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)
print(len(train_data))
print(train_data[:10])

wordsDict = reader._build_vocab(DATA_PATH+"/ptb.train.txt")

## k word ,v index
for k,v in wordsDict.items():
    if v == train_data[1]:
        print(k)


## print(train_data[1])
# ptb_producer返回的为一个二维的tuple数据。
result = reader.ptb_producer(train_data, 4, 5)
# 通过队列依次读取batch。
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(3):
        x, y = sess.run(result)
        print ("X%d: "%i, x)
        print ("Y%d: "%i, y)
    coord.request_stop()
    coord.join(threads)