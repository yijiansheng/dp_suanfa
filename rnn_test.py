import numpy as np


## x序列
X = [1,2]
## n向量
state = [0.0, 0.0]
## n*n
w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
## 给输入的x向量，准备n个维度
## 这样做，是为了hidden_out 和 w_out是同样的 n*1
w_cell_input = np.asarray([0.5, 0.6])
## w_out *1 = n*1
b_cell = np.asarray([0.1, -0.1])

w_output = np.asarray([[1.0], [2.0]])
b_output = 0.1


for i in range(len(X)):
    ## U * xt + W*st-1 这里的dot是矩阵相乘
    before_activation = np.dot(state, w_cell_state) \
                        + X[i] * w_cell_input \
                        + b_cell
    print(np.dot(state, w_cell_state))
    ## 激活
    state = np.tanh(before_activation)
    ## V * state
    final_output = np.dot(state, w_output) + b_output
    print ("before activation: ", before_activation)
    print ("state: ", state)
    print ("output: ", final_output)