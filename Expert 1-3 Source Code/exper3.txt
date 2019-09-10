
"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
# part1
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow as tf

# 载入数据
mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data", one_hot=True)

# 构建单层神经网络
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b

# 定义损失函数和优化器
y_ = tf.placeholder(tf.float32, [None, 10])

# The raw formulation of cross-entropy,
#
#   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
#                                 reduction_indices=[1]))
#
# can be numerically unstable.
#
# So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
# outputs of 'y', and then average across the batch.
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# 注意learning_rate
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# 训练模型:range内迭代次数
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i%100==0:
        print("cross_entropy error:",sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys}))

# 测试训练好的模型
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("test accuracy: ",sess.run(accuracy, feed_dict={x: mnist.test.images,
                                  y_: mnist.test.labels}))

# part2 ：选择图片测试
# 第几张图片？
p = 0

s = sess.run(y,feed_dict={x: mnist.test.images[p].reshape(1,784)})
print("Prediction : ",sess.run(tf.argmax(s, 1)))

#显示图片
plt.imshow(mnist.test.images[p].reshape(28,28), cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()