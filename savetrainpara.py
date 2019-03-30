#coding=utf-8

import tensorflow as tf
import numpy as np

'''
保存训练好网络的参数并读取参数
'''

# save to file
# remember to define the same dtype and shape when restore

"""
W = tf.Variable([[1,2,3],[3,4,5]], dtype=tf.float32, name='weights')
b = tf.Variable([[1,2,3]], dtype=tf.float32, name='biases')

init = tf.initialize_all_variables()

saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(init)
	save_path = saver.save(sess,"netpara/para.ckpt")
	print("Save to path: ", save_path)
"""
# restore variable
# redefine the same shape anf=d same type for your variable
W = tf.Variable(np.arange(6).reshape((2,3)), dtype=tf.float32, name="weights")
b = tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32, name="biases")
 
# not need init step
saver = tf.train.Saver()
with tf.Session() as sess:
	saver.restore(sess,"netpara/para.ckpt")
	print("weights:", sess.run(W))
	print("biases:", sess.run(b))