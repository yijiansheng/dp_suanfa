import re
import random
import ast
import itertools
import pickle
import numpy as np


## 测试用少量数据
train_data_file = "dataset/comprehen/cbtest_NE_valid_2000ex.txt"
valid_data_file = "dataset/comprehen/cbtest_NE_valid_2000ex.txt"


def preprocess_data(data_file, out_file):
    # stories[x][0]  tories[x][1]  tories[x][2]
    stories = []
    with open(data_file) as f:
        story = []
        for line in f:
            line = line.strip()
            if not line:
                story = []
            else:
                _, line = line.split(' ', 1)
                if line:
                    if '\t' in line:
                        q, a, _, answers = line.split('\t')
                        # tokenize
                        q = [s.strip() for s in re.split('(\W+)+', q) if s.strip()]
                        ## stories的形式是 (story[list],question,answer )
                        stories.append((story, q, a))
                    else:
                        line = [s.strip() for s in re.split('(\W+)+', line) if s.strip()]
                        story.append(line)

    samples = []
    for story in stories:
        story_tmp = []
        content = []
        for c in story[0]:
            content += c
        story_tmp.append(content)
        story_tmp.append(story[1])
        story_tmp.append(story[2])
        ## 相当于做了一次变化，将内容作为[0] question作为1，answer作为2
        samples.append(story_tmp)
    random.shuffle(samples)
    print(len(samples))


    with open(out_file, "w") as f:
        for sample in samples:
            f.write(str(sample))
            f.write('\n')

# preprocess_data(valid_data_file, 'dataset/comprehen/train.data')
# preprocess_data(valid_data_file, 'dataset/comprehen/valid.data')



## 检验是否正常的格式
def read_data(data_file):
    stories = []
    with open(data_file) as f:
        for line in f:
            line = ast.literal_eval(line.strip())
            stories.append(line)
    return stories


stories = read_data('dataset/comprehen/train.data') + read_data('dataset/comprehen/valid.data')

## 最长篇文章
content_length = max([len(s) for s, _, _ in stories])
## 最长的问题
question_length = max([len(q) for _, q, _ in stories])
print(content_length, question_length)



vocab = sorted(set(itertools.chain(*(story + q + [answer] for story, q, answer in stories))))
vocab_size = len(vocab) + 1
## 10501
print(vocab_size)
## { 'a':1 }
word2idx = dict((w, i + 1) for i, w in enumerate(vocab))



## 后面那个0，代表ASC协议
pickle.dump((word2idx, content_length, question_length, vocab_size), open('dataset/comprehen/vocab.data', "wb"),0)



##补充
def pad_sequences(sequences,
                  maxlen=None,
                  dtype='int32',
                  padding='post',
                  truncating='post',
                  value=0.):
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


# 转为向量
def to_vector(data_file, output_file):
    word2idx, content_length, question_length, _ = pickle.load(open('dataset/comprehen/vocab.data', "rb"))

    X = []
    Q = []
    A = []
    with open(data_file) as f_i:
        for line in f_i:
            line = ast.literal_eval(line.strip())
            x = [word2idx[w] for w in line[0]]
            q = [word2idx[w] for w in line[1]]
            a = [word2idx[line[2]]]

            X.append(x)
            Q.append(q)
            A.append(a)

    X = pad_sequences(X, content_length)
    Q = pad_sequences(Q, question_length)

    with open(output_file, "w") as f_o:
        for i in range(len(X)):
            f_o.write(str([X[i].tolist(), Q[i].tolist(), A[i]]))
            f_o.write('\n')



## 补零，使得content ,question , answer 都是 [800,100,1]这样的list，不到100的，后面补0
to_vector('dataset/comprehen/train.data', 'dataset/comprehen/train.vec')
to_vector('dataset/comprehen/valid.data', 'dataset/comprehen/valid.vec')

