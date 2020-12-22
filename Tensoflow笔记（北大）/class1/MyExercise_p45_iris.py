# -*- coding: UTF-8 -*-

#导入工具包
import tensorflow as tf
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np

#数据导入
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

#数据乱序
# seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样（为方便教学，以保每位同学结果一致）
np.random.seed(116)  # 使用相同的seed，保证输入特征和标签一一对应
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

#数据切分为训练集和测试集
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]
#数据类型转化
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)
#标签对应
train_db = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

#定义训练网络
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))
#学习率、训练轮数以及可视化需要用到的容器定义
lr = 0.1
epoch = 500
train_loss_results = []
test_acc = []
loss_all = 0

#迭代训练参数（核心部分）
for epoch in range(epoch):
    for step, (x_data, y_data) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w1) + b1
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(y_train, depth=3)
            loss = tf.reduce_mean(tf.square(y_ - y))
            loss_all += loss.numpy()

        grads = tape.gradient(loss,[w1,b1])

        w1.assign_sub(lr*grads[0])
        b1.assign_sub(lr*grads[1])

    print("Epoch {}, loss: {}".format(epoch, loss_all/4))
    train_loss_results.append(loss_all / 4)
    loss_all = 0




#测试
    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)
        pred = tf.cast(pred, dtype=y_test.dtype)
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_number += x_test.shape[0]

    acc = total_correct / total_number
    test_acc.append(acc)
    print("Test_acc:", acc)
    print("--------------------------")
#可视化
plt.title('LossFunctionCurve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss_results, label = "$Loss$")
plt.legend()
plt.show()

plt.title('AccCurve')#图片标题
plt.xlabel('Epoch')#x轴变量名称
plt.ylabel('Acc')#y轴变量名称
plt.plot(test_acc,label="$Accuracy$")#逐点画出test_acc值并连线，连线图标是Accuracy
plt.legend()
plt.show()