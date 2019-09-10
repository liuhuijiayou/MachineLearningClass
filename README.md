----------

# 目录 #

-	[准备工作](#准备工作)

-	[实验一、Python机器学习入门：有监督学习](#实验一python机器学习入门有监督学习)

	-	[实验目标](#实验目标)

	-	[实验器材及准备](#实验器材及准备)

	-	[实验内容与步骤](#实验内容与步骤)

	-	[实验结果](#实验结果)

	-	[课后习题](#课后习题)

-	[实验二、Python机器学习入门：无监督学习](#实验二python机器学习入门无监督学习)

	-	[实验目标](#实验目标-1)

	-	[实验器材及准备](#实验器材及准备-1)

	-	[实验内容与步骤](#实验内容与步骤-1)

	-	[实验结果](#实验结果-1)

	-	[课后习题](#课后习题-1)

-	[实验三、Python深度学习入门：单层神经网络](#实验三python深度学习入门单层神经网络)

	-	[实验目标](#实验目标-2)

	-	[实验器材及准备](#实验器材及准备-2)

	-	[实验内容与步骤](#实验内容与步骤-2)

	-	[实验结果](#实验结果-2)

	-	[课后习题](#课后习题-2)

-	[实验四、Python深度学习入门：人脸识别实验](#实验四python深度学习入门人脸识别实验)

	-	[实验目标](#实验目标-3)

	-	[实验器材及准备](#实验器材及准备-3)

	-	[实验内容与步骤](#实验内容与步骤-3)

		-	[实验环境及数据准备](#实验环境及数据准备)

		-	[从照片中获取人脸](#从照片中获取人脸)

		-	[获取脸部特征并进行仿射变换](#获取脸部特征并进行仿射变换)

		-	[获取面部特征编码文件](#获取面部特征编码文件)

		-	[进行完整的人脸识别实验](#进行完整的人脸识别实验)

-	[Jupyter环境部署](Jupyter)

	-	[Anaconda安装](Jupyter#anaconda安装)

	-	[TensorFlow安装](Jupyter#tensorflow安装)

-	[Docker环境部署](Docker)

	-	[Docker安装](Docker#docker安装)

	-	[Docker加速器](Docker#docker加速器)

	-	[Docker从文件载入镜像](Docker#docker从文件载入镜像)

	-	[Docker运行命令](Docker#docker运行命令)

----------

# 准备工作 #

-	开机选择第一个操作系统：`LINUX`

-	LINUX系统管理员密码：`123456`

-	系统启动后打开左边栏`FireFox`火狐浏览器![](http://www.firefox.com.cn/media/img/firefox/favicon.e6bb0e59df3d.ico)

-	在上方地址编辑栏输入`FTP`网址：

	>地址：[`ftp://10.105.240.91`](ftp://student:asdf1234@10.105.240.91/Machine%20Learning)

	>用户名：`student`

	>密码：`asdf1234`

-	进入`Machine Learning`目录

-	下载实验讲义PPT、实验指导手册及相关实验源代码文档

----------

# 实验一、Python机器学习入门：有监督学习 #

## 实验目标 ##

-	了解机器学习的基本概念，了解机器学习的应用方法。

-	通过实验掌握机器学习预测任务的基本流程。

## 实验器材及准备 ##

### 实验器材 ###

-	硬件：电脑PC一台

-	软件：Ubuntu、Anaconda3 5.0.1、Scikit-learn 0.19及其依赖包

### 实验准备 ###

-	查阅机器学习-有监督学习[基本原理和算法](http://sklearn.apachecn.org/#/docs/1)

-	查阅SVM分类器[基本原理](http://sklearn.apachecn.org/#/docs/5)

## 实验内容与步骤 ##

### Jupyter Notebook 简介 ###

>Jupyter Notebook，以前又称为IPython notebook，是一个交互式笔记本，支持运行40+种编程语言，详见[介绍](https://www.zhihu.com/question/37490497)。

### 打开编译环境 ###

1.	进入Ubuntu系统后，同时按下`Ctrl`+<font color=#A52A2A>`Alt`</font>+`T`打开终端`Terminal`窗口。

1.	在终端窗口中输入以下<font color=#A52A2A>**`1`**</font>条命令打开Jupyter Notebook编译环境：

	```bash
	jupyter notebook
	```

1.	键入`Enter`回车键后等待，浏览器会自动打开如下地址：

	![](https://i.imgur.com/PZYQqSc.png)

1.	点击页面右上方区域按钮`New`->`Python3`。

1.	将`exper1.txt`文件中代码复制入`In[]:`后光标中，键入`Shift`+`Enter`运行。

	![](https://i.imgur.com/PFyjMn9.png)

1.	代码文件`exper1.txt`中实现了以下步骤：
	1. 导入试验依赖模块
	   ```python
    	#导入matplotlib绘图工具包
    	import matplotlib.pyplot as plt
		# Import datasets, classifiers and performance metrics
       from sklearn import datasets, svm, metrics
	   ```
	1. 载入示例数据集。载入Scikit-learn自带数据集手写数字识别集（Handwritten Digits Data Set）
	
	   ```python
	   # The digits dataset
	   # 加载数据集
	   digits = datasets.load_digits()
	   ```
	1. 查看数据集。使用`matplotlib`显示数据集图片。
	
	   ```python
	   # The data that we are interested in is made of 8x8 images of digits, let's
	   # have a look at the first 4 images, stored in the `images` attribute of the
	   # dataset.  If we were working from image files, we could load them using
	   # matplotlib.pyplot.imread.  Note that each image must have the same size. For these
	   # images, we know which digit they represent: it is given in the 'target' of
	   # the dataset.
	   # 查看数据集前4张图片
	   images_and_labels = list(zip(digits.images, digits.target))
	   for index, (image, label) in enumerate(images_and_labels[:4]):
	       plt.subplot(2, 4, index + 1)
	       plt.axis('off')
	       plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
	       plt.title('Training: %i' % label)
	   ```
	
	1. 数据预处理。使用`numpy`将图片展开成向量。
	
	   ```python
	   # To apply a classifier on this data, we need to flatten the image, to
	   # turn the data in a (samples, feature) matrix:
	   # 数据预处理：将数据集展开成向量
	   n_samples = len(digits.images)
	   data = digits.images.reshape((n_samples, -1))
	   ```
	
	1. 构建分类器模型。使用Scikit-learn中的分类器`SVM`。
	
	   ```python
	   # Create a classifier: a support vector classifier
	   # 构建分类器SVM
	   classifier = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
	     degree=3, gamma=0.001, kernel='rbf',
	     max_iter=-1, probability=False, random_state=None, shrinking=True,
	     tol=0.001, verbose=False)
	   ```
	
	1. 训练分类器模型。使用一半数据集进行模型的训练。
	
	   ```python
	   # We learn the digits on the first half of the digits
	   # 训练分类器
	   classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])
	   ```
	1. 使用训练好的分类器模型预测另一半数据集。
	
	   ```python
	   # Now predict the value of the digit on the second half:
	   # 测试分类效果
	   expected = digits.target[n_samples // 2:]
	   predicted = classifier.predict(data[n_samples // 2:])
	   ```
	1. 检查分类器的预测效果。使用Scikit-learn自带`metrics`检查预测准确率、召回率及混淆矩阵（Confusion Matrix）等。
	
	   ```python
	   print("Classification report for classifier %s:\n%s\n"
	         % (classifier, metrics.classification_report(expected, predicted)))
	   print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
	   
	   images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
	   for index, (image, prediction) in enumerate(images_and_predictions[:4]):
	       plt.subplot(2, 4, index + 5)
	       plt.axis('off')
	       plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
	       plt.title('Prediction: %i' % prediction)
	   plt.show()
	   ```

## 实验结果 ##

-	分类器说明和每个分类的准确率`precision`，召回率`recall`，F1分数`f1-score`和各类别参与训练的样本数。

	![](https://i.imgur.com/PdUhY7M.png)

-	混淆矩阵：可看到测试数据集被分类的情况。

	![](https://i.imgur.com/mCVubl6.png)

-	训练和测试情况。

	![](https://i.imgur.com/2G57l0P.png)

## 课后习题 ##

-	参考[SVM参数表](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)修改SVM参数，如惩罚因子`C`、rbf核函数的系数`gamma`等，观察预测结果的变化情况。修改代码后键入`Shift`+`Enter`可再次运行。

	![](https://i.imgur.com/f55ZeJC.png)

----------

**附**`expert1.txt`文件内容如下：

```python
print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

# The digits dataset
# 加载数据集
digits = datasets.load_digits()

# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
# 查看前4张图片
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
# 数据预处理：展开成向量
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
# 构建分类器SVM
classifier = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  degree=3, gamma=0.001, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)

# We learn the digits on the first half of the digits
# 训练分类器
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

# Now predict the value of the digit on the second half:
# 测试分类效果
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()
```

----------

# 实验二、Python机器学习入门：无监督学习 #

## 实验目标 ##

-	了解机器学习有监督和无监督的区别。

-	通过实验掌握简单无监督算法使用方式。

## 实验器材及准备 ##

### 实验器材 ###

-	硬件：电脑PC一台

-	软件：Ubuntu、Anaconda3 5.0.1、Scikit-learn 0.19及其依赖包

### 实验准备 ###

-	查阅机器学习-无监督学习[基本原理和相关算法](http://sklearn.apachecn.org/#/docs/19)

-	查阅聚类基本原理及K-Means[算法原理](http://sklearn.apachecn.org/#/docs/22)

## 实验内容与步骤 ##

1.	打开编译环境。如实验一打开Jupyter Notebook，新建`New`->`Python3`交互窗口。

2. 代码文件`exper2.txt`中实现了以下步骤，详见**代码注释**：

	1.	载入示例数据集。载入Scikit-learn自带数据集手写数字识别集（Handwritten Digits Data Set）。
	
	1.	数据预处理。使用`numpy`将图片展开成向量。
	
	1.	学习和预测。使用Scikit-learn中的聚类算法`K-Means`，在全部数据集上做聚类。 
	
	1.	检查聚类效果。使用Scikit-learn自带`metrics`检查聚类效果。
	
	1.	聚类可视化。使用`matplotlib`可视化聚类结果（`PCA`降维到`2`维以便平面显示）。

## 实验结果 ##

-	数据集包含10个分类（手写数字1-10），1797个样本，特征维度为64维。

	![](https://i.imgur.com/prIQJYS.png)

-	可以看到3个不同kmeans初始化中心点方法的聚类器的效果，注意使用PCA-based方法初始化中心点速度极快，因为中心点更新次数少。

	![](https://i.imgur.com/fYW8F6h.png)

	![](https://i.imgur.com/Ypn5CuK.png)

-	最后在图中可以看到PCA降维到2维的数据聚类情况。

	![](https://i.imgur.com/6hPIppJ.png)

## 课后习题 ##

-	参考[K-Means参数表](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)修改K-Means参数，如**分类器**初始化类型（`k-means++`、`random`、`PCA-based`）、类别数`n_clusters`等，观察聚类结果变化情况。

	![](https://i.imgur.com/eEP75zg.png)

----------

**附**`expert2.txt`文件内容如下：

```python
print(__doc__)

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# 设定随机数种子
np.random.seed(42)

# 加载数据集
digits = load_digits()
data = scale(digits.data)

# 解析数据集
n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

sample_size = 300

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))


print(82 * '_')
print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')

# 函数：训练并测试分类效果
def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
	  % (name, (time() - t0), estimator.inertia_,
	     metrics.homogeneity_score(labels, estimator.labels_),
	     metrics.completeness_score(labels, estimator.labels_),
	     metrics.v_measure_score(labels, estimator.labels_),
	     metrics.adjusted_rand_score(labels, estimator.labels_),
	     metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
	     metrics.silhouette_score(data, estimator.labels_,
				      metric='euclidean',
				      sample_size=sample_size)))

# 构建K-means分类器1，传入以上函数
bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
	      name="k-means++", data=data)

# 构建K-means分类器2，传入以上函数
bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
	      name="random", data=data)

# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
# 构建K-means分类器3，添加PCA降维，传入以上函数
pca = PCA(n_components=n_digits).fit(data)
bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
	      name="PCA-based",
	      data=data)
print(82 * '_')

# #############################################################################
# 聚类可视化。使用matplotlib可视化聚类结果（PCA降维到2维以便平面显示）
# Visualize the results on PCA-reduced data

reduced_data = PCA(n_components=2).fit_transform(data)
# 习题：修改此处参数
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
	   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
	   cmap=plt.cm.Paired,
	   aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
	    marker='x', s=169, linewidths=3,
	    color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
	  'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
```

----------

# 实验三、Python深度学习入门：单层神经网络 #

## 实验目标 ##

-	了解深度学习的基本概念。

-	通过实验学会使用框架实现简单神经网络。

## 实验器材及准备 ##

### 实验器材 ###

-	硬件：电脑PC一台

-	软件：Ubuntu、Anaconda3 5.0.1、TensorFlow 1.3.0及其依赖包

### 实验准备 ###

-	查阅深度学习与TensorFlow[基本原理和相关算法](https://www.tensorflow.org/get_started/mnist/beginners)

## 实验内容与步骤 ##

1.	打开编译环境。如实验一打开Jupyter Notebook，新建`New`->`Python3`交互窗口。

1.	代码文件`exper3.txt`中实现了以下步骤，详见**代码注释**：

	-	首先粘贴`part1`部分代码，运行`part1`部分代码（可能要运行几分钟之后才有结果输出，耐心等待一下）：
	
		1.	载入示例数据集。载入Tensorflow自带数据集手写数字识别集（MNIST Data）。
	
		1.	构建神经网络。利用Tensorflow构建简单神经网络，定义损失函数和优化器。
	
		1.	学习和预测。在给定的训练样本上运行以上神经网络，观察交叉熵（Cross Entropy）误差`error`值的变化，并在待测样本上做出预测。 
	
		1.	检查预测效果。定义准确率`accuracy`计算方式，检查预测准确率。
	
	-	在下一个输入框中，运行`part2`部分代码，查看预测结果和图片：
	
		1.	选择图片并进行预测。选择一张图片，用训练好的模型对新的图片进行预测。
	
		1.	输出图片及预测结果。使用`matplotlib`可视化图及结果。

## 实验结果 ##

-	随着迭代的进行，神经网络在数据集上的交叉熵（Cross Entropy）误差`error`值越来越小，代表正在慢慢拟合训练数据，最后在测试集上的测试准确率`accuracy`为`91.02%`。

	![](https://i.imgur.com/Igv7sT4.png)

-	选取测试集图片，进行预测：

	![](https://i.imgur.com/bnho3Gf.png)

## 课后习题 ##

-	修改学习迭代次数`range`、学习率`learning_rate`等，观察结果的变化。

	![](https://i.imgur.com/cI6Lehb.png)

----------

**附**`expert3.txt`文件内容如下：

`part1`:

```python
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
```

`part2`:

```python
# part2 ：选择图片测试
# 第几张图片？
p = 0

s = sess.run(y,feed_dict={x: mnist.test.images[p].reshape(1,784)})
print("Prediction : ",sess.run(tf.argmax(s, 1)))

#显示图片
plt.imshow(mnist.test.images[p].reshape(28,28), cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
```

----------

# 实验四、Python深度学习入门：人脸识别实验 #

## 实验目标 ##

-	了解人脸识别的基本原理

-	通过实验熟悉人脸识别的四个过程

## 实验器材及准备 ##

### 实验器材 ###

-	硬件：电脑PC一台

-	软件：Ubuntu、Docker、openface项目Docker容器镜像及其相关依赖包

### 实验准备 ###

-	仔细阅读课程讲义PPT内容，了解人脸识别**四个基本步骤**。

-	查阅Docker工具[使用手册](http://www.docker.org.cn/book/docker/what-is-docker-16.html)

-	查阅机器学习及人脸识别相关理论及算法：[HOG](http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)、[仿射变换](https://en.wikipedia.org/wiki/Affine_transformation)、[128维embedding面部特征向量编码](https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf)

-	查阅openface相关背景知识：[cmusatyalab](https://cmusatyalab.github.io/openface/)、[openface](https://github.com/cmusatyalab/openface)

## 实验内容与步骤 ##

### 实验环境及数据准备 ###

#### Linux相关基础知识 ####

1.	本实验说明中所有命令语句均可使用 **复制**`Copy` / **粘贴**`Paste` 操作在实验机的`Terminal`命令窗口中直接运行，调取命令窗口 **快捷键** 为`Ctrl`+<font color=#A52A2A>`Alt`</font>+`T`。

1.	**复制**`Copy` 命令语句时，请用鼠标选定本手册中每条命令语句的 **第一个非空格字符** 直至 **最后一个非空格字符**，注意命令语句中不要遗漏 **斜杠**`/` 或者 **空格**`Space`。

1.	`Terminal`命令窗口中的 **复制**`Copy` 操作可以使用 **鼠标右键菜单->复制**`Copy` 或者使用 **快捷键**`Ctrl`+<font color=#A52A2A>`Shift`</font>+`C`。

1.	`Terminal`命令窗口中的 **粘贴**`Paste` 操作可以使用 **鼠标右键菜单->粘贴**`Paste` 或者使用 **快捷键**`Ctrl`+<font color=#A52A2A>`Shift`</font>+`V`。

1.	`Terminal`命令窗口中可以使用 **方向键** **上**`∧` / **下**`∨` 查看之前自己输入过的命令语句。

1.	`Terminal`命令窗口中输入命令或路径时，可以使用 **快捷键**`Tab` 对命令或路径进行快速补全操作。

#### Docker简介 ####

>Docker是一个开源的引擎，可以轻松的为任何应用创建一个轻量级的、可移植的、自给自足的容器。开发者在笔记本上编译测试通过的容器可以批量地在生产环境中部署，包括VMs（虚拟机）、[bare metal](http://www.whatis.com.cn/word_5275.htm)、OpenStack集群和其他的基础应用平台。

1.	本实验利用基于openface开源项目所提供的Docker容器镜像`bamos/openface`环境进行人脸识别实验。

1.	<font color=#A52A2A>**注意事项**</font>【<font color=#A52A2A>`！！！重要！！！`</font>】：

	在`Terminal`命令窗口中进入Docker容器内运行的所有实验代码的**工作目录**均为`/root/openface`，请把命令语句中**所有的** `your_test_image_fullpath.jpg`替换为你自己的**完整图片路径**，例如：`/home/bupt/my_pic.jpg`

#### 运行Docker实验环境 ####

1.	使用快捷键`Ctrl`+<font color=#A52A2A>`Alt`</font>+`T`打开`Terminal`命令行窗口

1.	在`Terminal`命令行窗口中依次运行以下<font color=#A52A2A>**`2`**</font>条命令进入Docker容器openface环境内：

	```bash
	sudo xhost +local:root
	```

	```bash
	sudo docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /home/$USER:/home/$USER:rw -t -i openface/allset /bin/bash
	```

1.	运行以上第<font color=#A52A2A>`1`</font>条命令后会被要求输入管理员密码，密码为<font color=#A52A2A>`123456`</font>，在输入密码时`Terminal`命令行窗口中<font color=#A52A2A>**不会显示密码输入**</font>，在确认输入无误后单击回车`Enter`按钮即可，若显示如下信息则表示第<font color=#A52A2A>`1`</font>条命令运行成功，可以继续输入第<font color=#A52A2A>`2`</font>条命令：

	![](https://i.imgur.com/ip3WtFF.png)

1.	运行完以上<font color=#A52A2A>**`2`**</font>条命令进入Docker容器后，运行以下<font color=#A52A2A>**`1`**</font>条命令转至 **openface工作目录**`/root/openface`【<font color=#A52A2A>**`！！！重要！！！`**</font>】：

	```bash
	cd /root/openface
	```

1.	运行以下<font color=#A52A2A>**`4`**</font>条命令清除示例样本文件（若提示文件不存在则可忽略）：

	```bash
	rm /root/openface/aligned_face_0.jpg
	```

	```bash
	rm -r /home/bupt/training-images
	```

	```bash
	rm -r /home/bupt/aligned-images
	```

	```bash
	rm -r /home/bupt/generated-embeddings
	```

#### 建立人脸样本库 ####

1.	运行以下<font color=#A52A2A>**`1`**</font>条命令（**或者**使用鼠标左键单击左边栏图标![](https://i.imgur.com/G2jr5vR.png)手动进入bupt用户主目录`/home/bupt`内，再使用鼠标右键菜单新建文件夹`New Folder`选项）创建`training-images`文件夹（若提示目录已存在则可忽略）：

	```bash
	mkdir -vm 777 /home/bupt/training-images
	```

1.	运行类似以下命令语句（**或者**使用鼠标左键双击文件夹图标![](https://i.imgur.com/r7fF9KI.png)进入`/home/bupt/training-images/`目录内，再使用鼠标右键菜单新建文件夹`New Folder`选项）创建各个（<font color=#A52A2A>**必须`2`个以上**</font>）不同人的样本库文件夹，把以下命令中的`person1`、`person2`、`person3`等各自改为你自定义的名称后再运行，例如你自己的名字、你朋友的名字、你喜欢的明星的名字等：

	```bash
	mkdir -vm 777 /home/bupt/training-images/person1
	```

	```bash
	mkdir -vm 777 /home/bupt/training-images/person2
	```

	```bash
	mkdir -vm 777 /home/bupt/training-images/person3
	```

	![](https://i.imgur.com/iXEOeL6.png)

1.	拷贝需要作为训练集的人脸照片样本至相应目录：

	通过网络下载或者U盘等移动存储设备把照片样本复制`Copy`至bupt用户主目录`/home/bupt`下的`training-images`文件夹中对应的各文件目录`person1`、`person2`、`person3`下（此时文件夹名应该**已经**替换为你自定义的名称了，并且每一个文件夹都<font color=#A52A2A>**必须**</font>要拷入图片文件使其<font color=#A52A2A>**不为空**</font>，如果不小心多建了不用的文件夹请<font color=#A52A2A>**务必`Delete`删除**</font>）。
	
	![](https://i.imgur.com/WJRBj5Y.png)

1.	<font color=#A52A2A>**注意事项**</font>【<font color=#A52A2A>`！！！重要！！！`</font>】：

	-	把各个人的照片样本拷贝至<font color=#A52A2A>**与其相应**</font>的样本库文件夹，并保证每个照片样本中<font color=#A52A2A>**只有一张人脸**</font>，并清晰可见（<font color=#A52A2A>**眉/眼/鼻/嘴/脸廓**</font>完整，最好<font color=#A52A2A>**不要**</font>佩戴眼镜）。

	-	<font color=#A52A2A>**必须确保**</font>每个已建立的人脸库文件夹中均包含照片样本而<font color=#A52A2A>**不为空**</font>。
	
	-	<font color=#A52A2A>**必须至少**</font>建立<font color=#A52A2A>**`2`**</font>个以上的人脸库文件夹，推荐<font color=#A52A2A>`3`</font>个以上。

	-	推荐每个人都有<font color=#A52A2A>**`10`**</font>张以上的照片样本，最好包含不同角度、侧面等，但是必须保证<font color=#A52A2A>**眉/眼/鼻/嘴/脸廓**</font>完整。

	-	<font color=#A52A2A>**并不需要**</font>提供经过align裁剪后的照片样本，普通的日常照片就行，openface会根据命令对照片样本进行align裁剪。

----------

### 从照片中获取人脸 ###

#### 运行`step-1_find-faces.py`获取人脸位置 ####

1.	把以下<font color=#A52A2A>**`1`**</font>条命令中的`your_test_image_fullpath.jpg`替换为你自己准备的待测图片包含**文件名及全路径**的<font color=#A52A2A>**完整路径**</font>，例如：`/home/bupt/my_pic.jpg`，再运行命令：

	```bash
	python /root/openface/step-1_find-faces.py your_test_image_fullpath.jpg
	```

1.	运行以上命令之后会显示如下类似图片结果：

	![](https://i.imgur.com/3Lg1NTx.jpg)

1.	在Terminal命令行窗口中<font color=#A52A2A>**键入`Enter`回车按钮**</font>继续。

----------

### 获取脸部特征并进行仿射变换 ###

#### 运行`step-2a_finding-face-landmarks.py`获取脸部特征 ####

1.	把以下<font color=#A52A2A>**`1`**</font>条命令中的`your_test_image_fullpath.jpg`替换为你自己准备的待测图片包含**文件名及全路径**的<font color=#A52A2A>**完整图片路径**</font>，例如：`/home/bupt/my_pic.jpg`，再运行命令：

	```bash
	python /root/openface/step-2a_finding-face-landmarks.py your_test_image_fullpath.jpg
	```

1.	运行以上命令之后会显示如下类似图片结果：

	![](https://i.imgur.com/MQq0N4u.jpg)

1.	在Terminal命令行窗口中<font color=#A52A2A>**键入`Enter`回车按钮**</font>继续。

#### 运行`step-2b_projecting-faces.py`获取仿射变换后的照片 ####

1.	把以下<font color=#A52A2A>**`1`**</font>条命令中的`your_test_image_fullpath.jpg`替换为你自己准备的待测图片包含**文件名及全路径**的<font color=#A52A2A>**完整图片路径**</font>，例如：`/home/bupt/my_pic.jpg`，再运行命令：

	```bash
	python /root/openface/step-2b_projecting-faces.py your_test_image_fullpath.jpg
	```

1.	运行以上命令之后会在工作目录`/root/openface`下生成如下相应的裁剪图片文件`aligned_face_0.jpg`：

1.	可运行以下<font color=#A52A2A>**`1`**</font>条命令把裁剪后的图片文件`aligned_face_0.jpg`拷贝至bupt用户主目录`/home/bupt`：

	```bash
	cp /root/openface/aligned_face_0.jpg /home/bupt/
	```

	然后在主目录**双击**打开![](https://i.imgur.com/4Vn1zKX.png)查看：

	![](https://i.imgur.com/1Y9wwsz.png)

----------

### 获取面部特征编码文件 ###

#### 运行`main.lua`对仿射变换后的人脸图片提取特征编码 ####

1.	依次运行以下<font color=#A52A2A>**`3`**</font>条命令：

	```bash
	mkdir -p /home/bupt/my_aligned_face/my_face
	```

	```bash
	cp /root/openface/aligned_face_0.jpg /home/bupt/my_aligned_face/my_face/
	```

	```bash
	/root/openface/batch-represent/main.lua -outDir /home/bupt/my_reps/ -data /home/bupt/my_aligned_face/
	```

1.	运行以上命令之后可在`/home/bupt/my_reps`目录下找到如下相应的`128`维面部特征编码文件`reps.csv`：

	![](https://i.imgur.com/RsTyaaW.png)

	可**双击**打开![](https://i.imgur.com/ozCFHGi.png)查看文件内容：

	![](https://i.imgur.com/rmYCERD.png)

----------

### 进行完整的人脸识别实验 ###

#### 运行`align-dlib.py`进行仿射变换 ####

1.	运行以下<font color=#A52A2A>**`1`**</font>条命令：

	```bash
	/root/openface/util/align-dlib.py /home/bupt/training-images/ align outerEyesAndNose /home/bupt/aligned-images/ --size 96
	```

1.	运行以上命令之后可在`/home/bupt/aligned-images`目录下找到仿射变换后的图片文件：

	![](https://i.imgur.com/h08HjNS.png)

	![](https://i.imgur.com/jTfoN0m.png)

#### 运行`main.lua`获取`128`维面部特征向量表示文件 ####

1.	运行以下<font color=#A52A2A>**`1`**</font>条命令：

	```bash
	/root/openface/batch-represent/main.lua -outDir /home/bupt/generated-embeddings/ -data /home/bupt/aligned-images/
	```

1.	运行以上命令之后可在`/home/bupt/generated-embaddings`目录下找到`labels.csv`特征向量标识文件及`reps.csv`特征向量表示文件：

	![](https://i.imgur.com/kHfR4tD.png)

#### 运行`classifier.py train`训练样本集并生成分类器 ####

1.	运行以下<font color=#A52A2A>**`1`**</font>条命令：

	```bash
	/root/openface/demos/classifier.py train /home/bupt/generated-embeddings/
	```

1.	运行以上命令之后可在`/home/bupt/generated-embaddings`目录下找到如下`classifier.pkl`分类器文件：

	![](https://i.imgur.com/bh7Y5ev.png)

#### 运行`classifier.py infer`识别被测照片 ####

1.	把以下<font color=#A52A2A>**`1`**</font>条命令中的`your_test_image_fullpath.jpg`替换为你自己准备的待测图片包含**文件名及全路径**的<font color=#A52A2A>**完整图片路径**</font>，例如：`/home/bupt/my_pic.jpg`，再运行命令：

	```bash
	/root/openface/demos/classifier.py infer /home/bupt/generated-embeddings/classifier.pkl your_test_image_fullpath.jpg
	```

1.	运行以上命令之后会在`Terminal`命令窗口中显示类似如下识别结果：

	![](https://i.imgur.com/MpfSDla.png)

----------
