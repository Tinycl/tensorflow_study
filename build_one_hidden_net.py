# coding = utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, activation_function=None):
	# add one more layer and return the output of this layer
	# in_size row , out_size col
	Weights = tf.Variable(tf.random_normal([in_size,out_size]))
	biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
	Wx_plus_b = tf.matmul(inputs,Weights) + biases
	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)
	return outputs
	
# make up soe real data
x_data = np.linspace(-1,1,300)[:,np.newaxis]
#随机噪声 方差0.05
noise = np.random.normal(0,0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# define placeholder for inputs to newwork
xs = tf.placeholder(tf.float32, [None,1])
ys = tf.placeholder(tf.float32, [None,1])
#输入层和输出层的维度是相同的，隐层可以自定义
# add hidden layer
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1,10,1,activation_function=None)
# loss , the error between prediction and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init) #一定要初始化变量

# plot the real data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion() # 不阻塞后面代码执行
plt.show()

for i in range(10000):
	sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
	if i % 50 == 0:
		# 打印loss， loss 要一直减小
		#print(sess.run(loss,feed_dict={xs:x_data, ys:y_data}))
		# 试图删线
		try:
			ax.lines.remove(lines[0])
		except Exception:
			pass
		prediction_value = sess.run(prediction,feed_dict={xs:x_data}) #读取预测值，与其有关的placeholder var
		lines = ax.plot(x_data,prediction_value,'r-',lw=5)
		plt.pause(0.1) # 0.1s 暂停一下