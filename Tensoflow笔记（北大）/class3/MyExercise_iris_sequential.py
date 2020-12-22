#六步法
#①导入相关包
#②声明相关的参数，说明测试集、训练集
#③定义网络结构sequential(神经元个数，激活函数，正则化)
#④说明相关方法compile
#⑤定义训练过程fit
#⑥打印网络结构和参数统计summary

import tensorflow as tf
from sklearn import datasets
import numpy as np

x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

print(y_train)

'''np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)

#说明神经元个数，激活函数，正则化   Dense表示全连接层
model = tf.keras.models.Sequential([tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())])

#优化器，损失函数，准确率
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

#validation_split(划分多少比列的训练集到测试集中去), validation_data(测试集的输入特征，测试集的标签), validation_freq(多少个epochs测试一次)
model.fit(x_train, y_train, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)

#
model.summary()'''