import tensorflow as tf


x = tf.Variable(tf.truncated_normal([1]), name="x")
## optimizer的意义是
##
goal = tf.pow(x-3,2, name="goal")

with tf.Session() as sess:
    x.initializer.run()
    print(x.eval())
    print(goal.eval())

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
## 用optimizer 降 goal
#train_step = optimizer.minimize(goal)


def simple_train():
    with tf.Session() as sess:
        x.initializer.run()
        for i in range(10):
            print ("x: ", x.eval())
            ## 训练一次
            train_step.run()
            print ("goal: ",goal.eval())
##simple_train()

## minimize() = compute_gradients() + apply_gradients()
## 其实这是两个步骤 找出来需要更新的参数 然后应用梯度
gra_and_var = optimizer.compute_gradients(goal)
#train_step = optimizer.apply_gradients(gra_and_var)

##simple_train()



## 控制梯度爆炸
## 形成解元组
gradients, vriables = zip(*optimizer.compute_gradients(goal))
gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
## 降 前面是梯度，后面是变量
train_step = optimizer.apply_gradients(zip(gradients, vriables))
#simple_train()




## 加入学习率
global_step = tf.Variable(0)
## 有衰减因子
learning_rate = tf.train.exponential_decay(3.0, global_step, 3, 0.3, staircase=True)
optimizer2 = tf.train.GradientDescentOptimizer(learning_rate)
gradients, vriables = zip(*optimizer2.compute_gradients(goal))
## 防止梯度爆炸
gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
train_step3 = optimizer2.apply_gradients(zip(gradients, vriables),global_step=global_step)



with tf.Session() as sess:
    global_step.initializer.run()
    x.initializer.run()
    for i in range(10):
        print ("x: ", x.eval())
        train_step3.run()
        ## goal尽可能减小，找到x的值
        print ("goal: ",goal.eval())