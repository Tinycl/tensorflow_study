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
		
#
matrix1 = tf.constant([[3,1]]) #1行2列
matrix2 = tf.constant([[2],[2]]) #2行1列
product = tf.matmul(matrix1,matrix2) #matrix multiply np.dot(m1,m2)

#session 需要手动close 
print("Session test")
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()
#session 自动close
with tf.Session() as sess:
	result2 = sess.run(product)
	print(result2)
	
#Variable 操作
print("Variable test")
state = tf.Variable(0, name='counter')#定义一个variable
print(state.name)
one = tf.constant(1) #定义一个常量
new_value = tf.add(state,one) #相加
update = tf.assign(state,new_value) #更新
init = tf.initialize_all_variables()
with tf.Session() as sess:
	sess.run(init)
	for _ in range(3):
		sess.run(update)
		print(sess.run(state))

#placeholder 类似占位变量，run时传入变量
print("placeholder test")
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1,input2)
with tf.Session() as sess:
	print(sess.run(output, feed_dict={input1:[7.0],input2:[2.]}))
	