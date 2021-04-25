from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.keras import Model, layers, datasets
import numpy as np
import tensorflow_datasets as tfds

# MNIST 数据集参数
num_classes = 10 # 所有类别（数字 0-9）

# 训练参数
learning_rate = 0.001
training_steps = 500
batch_size = 128
display_step = 10

# 准备MNIST数据

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images, test_images = np.array(train_images, np.float32) / 255.0, np.array(test_images,np.float32) / 255.0

train_data = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).repeat().shuffle(1000).batch(64)
#test_data = tf.data.Dataset.from_tensor_slices((test_images, test_labels))


# 构建网络
class BuildNet(Model):

    def __init__(self):
        #继承Model所有特性
        super(BuildNet, self).__init__()
        #第一次卷积，核的个数（输出维度），核大小是5，relu函数作为激活函数
        self.conv1 = layers.Conv2D(filters=32, kernel_size=3, activation=tf.nn.relu)
        #第一次池化，池化大小是（2,2），步长为（2,2）
        self.maxpool1 = layers.MaxPool2D(pool_size=2, strides=2)

        #第二次卷积和池化
        self.conv2 = layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu)
        self.maxpool2 = layers.MaxPool2D(pool_size=2, strides=2)

        #第三次卷积和池化
        self.conv3 = layers.Conv2D(filters=128, kernel_size=3, activation=tf.nn.relu)
        self.maxpool3 = layers.MaxPool2D(pool_size=2, strides=2)

        #数据展平
        self.flatten = layers.Flatten()
        #全连接层
        self.fc1 = layers.Dense(units=1024, activation=tf.nn.relu)
        self.fc2 = layers.Dense(units=512)
        #dropout 防止过拟合
        self.dropout = layers.Dropout(rate=0.5)

        #网络输出
        self.out = layers.Dense(num_classes)

    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 28, 28, 1])
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout(x, training=is_training)
        x = self.out(x)
        if not is_training:
            # tf 交叉熵接收没有经过softmax的概率输出，所以只有不是训练时才应用softmax
            x = tf.nn.softmax(x)
        return x
        
# 交叉熵损失
def cross_entropy_loss(x,y):
    y = tf.cast(y, tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    return tf.reduce_mean(loss)

# 准确率评估
def accuracy_y(y_pre, y_true):
    correct_pred = tf.equal(tf.argmax(y_pre, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32), axis=-1)

# 优化过程
def run_optimization(x, y, op, conv_net):
    with tf.GradientTape() as g:
        pred = conv_net(x, is_training=True)
        loss = cross_entropy_loss(pred, y)
    trainable_variables = conv_net.trainable_variables
    gradients = g.gradient(loss, trainable_variables)
    op.apply_gradients(zip(gradients, trainable_variables))

if __name__ == "__main__":
    conv_net = BuildNet()
    optimizer = tf.optimizers.Adam(learning_rate)
    for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps),1):
        #print(step, batch_x.shape)
        run_optimization(batch_x, batch_y, optimizer, conv_net)

        if step % display_step == 0:
            pred = conv_net(batch_x)
            loss = cross_entropy_loss(pred, batch_y)
            acc = accuracy_y(pred, batch_y)
            print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))
    
    pred = conv_net(test_images)
    print(pred[0:10], test_labels[0:10])
    print("Test Accuracy: %f" % accuracy_y(pred, test_labels))
    #model = tf.keras.Sequential(conv_net)
    #model.save('./model/ckpt/xxx.md')
    pass