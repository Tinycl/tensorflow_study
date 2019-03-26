# coding = utf-8
import tensorflow as tf
import numpy as np

#输入数据 y=ax+b
#预测0.1 和 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

#初始化a b
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))
#预测值
y = Weights*x_data + biases
#损失函数
loss = tf.reduce_mean(tf.square(y-y_data))
#优化损失函数 学习率0.5
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
#初始化变量
init = tf.initialize_all_variables()
#初始化session
sess = tf.Session()
sess.run(init)
#训练200次，每隔20次打印预测的值
for step in range(201):
	sess.run(train)
	if step % 20 == 0:
		print(step, sess.run(Weights), sess.run(biases))