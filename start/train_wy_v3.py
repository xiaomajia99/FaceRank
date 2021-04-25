from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.keras import Model, layers, datasets, models
from PIL import Image
from sklearn.metrics import r2_score

import numpy as np
import tensorflow_datasets as tfds
import os
import pandas as pd
import matplotlib.pyplot as plt



# MNIST 数据集参数
num_classes = 10 # 所有类别（数字 0-9）

# 训练参数
learning_rate = 0.001
batch_size = 300
display_step = 10
face_size = 128

TRAIN_DATA_DIR = "./data/resize_train_data/"
TEST_DATA_DIR = "./data/resize_test_data/"


# minist 数据集
def get_mnist_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    train_images, test_images = np.array(train_images, np.float32) / 255.0, np.array(test_images,np.float32) / 255.0

    rand_index = np.random.permutation(train_images.shape[0])
    rand_train_x = train_images[rand_index]
    rand_train_y = train_labels[rand_index]

    test_images = tf.reshape(test_images,[-1,28,28,1])
    x_valid, x_train = tf.reshape(rand_train_x[0:20000], [-1,28,28,1]), tf.reshape(rand_train_x[20000:],[-1,28,28,1])
    y_valid, y_train = rand_train_y[0:20000], rand_train_y[20000:]

    return x_train, y_train, x_valid, y_valid, test_images, test_labels

# 头像数据集
def get_face_data():
    train_images = os.listdir(TRAIN_DATA_DIR)
    test_images = os.listdir(TEST_DATA_DIR)
    train_images = [ image for image in train_images if 'AM' in image]
    test_images = [image for image in test_images if  'AM' in image]

    train_x = []
    train_y = []
    test_x = []
    test_y = []
    # 读取训练数据
    for image in train_images:
        score = float(image[0:image.find('-')])
        img = Image.open(TRAIN_DATA_DIR + image)
        img = np.array(img, np.float32)
        img = np.reshape(img, [face_size,face_size,3]) / 255.0
        train_x.append(img)
        train_y.append(score)
    # 读取测试数据
    for image in test_images:
        score = float(image[0:image.find('-')])
        img = Image.open(TEST_DATA_DIR + image)
        img = np.array(img, np.float32)
        img = np.reshape(img, [face_size,face_size,3]) / 255.0
        test_x.append(img)
        test_y.append(score)
    train_x, train_y = np.array(train_x, np.float32), np.array(train_y, np.float32)
    test_x, test_y = np.array(test_x, np.float32), np.array(test_y, np.float32)

    rand_index = np.random.permutation(train_x.shape[0])
    rand_train_x = train_x[rand_index]
    rand_train_y = train_y[rand_index]

    x_valid, x_train = tf.reshape(rand_train_x[0:batch_size], [-1,face_size,face_size,3]), tf.reshape(rand_train_x[batch_size:],[-1,face_size,face_size,3])
    y_valid, y_train = rand_train_y[0:batch_size], rand_train_y[batch_size:]

    return x_train, y_train, x_valid, y_valid, test_x, test_y

# 构建网络
def createModel():
    model = tf.keras.models.Sequential()
    #第一次卷积，核的个数（输出维度）32，核大小是3，relu函数作为激活函数, 是否包含边界，输入格式
    model.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=(28,28,1)))
    #第一次池化，池化大小是（2,2），步长为（2,2）
    model.add(layers.MaxPool2D(pool_size=2, strides=2))

    #第二次卷积和池化
    model.add(layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=2, strides=2))

    #数据展平
    model.add(layers.Flatten())

    #全连接层
    model.add(layers.Dense(units=512, activation='relu'))
    #dropout 防止过拟合
    model.add(layers.Dropout(rate=0.3))

    #网络输出
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])    
    return model

# 构建网络
def createFaceModel():
    model = tf.keras.models.Sequential()
    #第一次卷积，核的个数（输出维度）32，核大小是3，relu函数作为激活函数, 是否包含边界，输入格式
    model.add(layers.Conv2D(filters=32, kernel_size=3, activation=tf.nn.relu, padding='same', input_shape=(face_size,face_size,3)))
    #第一次池化，池化大小是（2,2），步长为（2,2）
    model.add(layers.MaxPool2D(pool_size=2, strides=2))
    #model.add(layers.AveragePooling2D(pool_size=2, strides=2))

    #第二次卷积和池化
    model.add(layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu, padding='same'))
    model.add(layers.MaxPool2D(pool_size=2, strides=2))
    #model.add(layers.AveragePooling2D(pool_size=2, strides=2))

    #第三次卷积和池化
    model.add(layers.Conv2D(filters=128, kernel_size=3, activation=tf.nn.relu, padding='same'))
    model.add(layers.MaxPool2D(pool_size=2, strides=2))
    #model.add(layers.AveragePooling2D(pool_size=2, strides=2))

    #数据展平
    model.add(layers.Flatten())

    #全连接层
    model.add(layers.Dense(units=2048, activation=tf.nn.relu))
    model.add(layers.Dropout(rate=0.3))
    model.add(layers.Dense(units=1024, activation=tf.nn.relu))
    model.add(layers.Dropout(rate=0.3))
    model.add(layers.Dense(units=1024, activation=tf.nn.relu))
    model.add(layers.Dropout(rate=0.3))
    model.add(layers.Dense(units=1024, activation=tf.nn.relu))

    #网络输出
    #model.add(layers.Dense(num_classes, activation='softmax'))
    model.add(layers.Dense(1))

    #model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])    
    model.compile(loss='mse', optimizer='adam')
    return model
                
# 可视化训练指标
def plot_data(history):
    x=history.epoch
    y_loss = history.history['loss']
    #y_acc = history.history['accuracy']
    y_val_loss = history.history['val_loss']
    #y_val_acc = history.history['val_accuracy']
    
    plt.figure()
    plt.plot(x,y_loss, 'r--', label='loss')
    #plt.plot(x,y_acc, 'g--', label='acc')
    plt.plot(x,y_val_loss, 'b--', label='val_loss')
    #plt.plot(x,y_val_acc, 'k--', label='val_acc')
    plt.xlabel('Iters')
    plt.ylabel('Rate')
    plt.title('Train')
    plt.legend()
    plt.show()

if __name__ == "__main__":

    x_train, y_train, x_valid, y_valid, x_test, y_test = get_face_data()

    #print(x_train.shape, y_train.shape, x_valid,shape, y_valid.shape, x_test.shape, y_test.shape)

    modelPath = './model/second/model.ckpt'
    modelDir = os.path.dirname(modelPath)
    #modelInitDir = './model/init/'
    modelInitDir = './model/male/'

    #model = createModel()
    #model = createFaceModel()
    model = models.load_model(modelInitDir)
    model.summary()


    #callback = tf.keras.callbacks.ModelCheckpoint(filepath = modelPath, save_weights_only=True, verbose=1)

    #history = model.fit(x_train, y_train, epochs=100, validation_data=(x_valid,y_valid), callbacks = [callback])
    #history = model.fit(x_train, y_train, epochs=30, validation_data=(x_valid,y_valid))
    

    model.evaluate(x_test, y_test)
    y_pred = model.predict(x_test)
    print(r2_score(y_test, y_pred))

    emb = model.layers[0]
    weights = emb.get_weights()[0]
    print(weights.shape, len(emb.get_weights()), emb.get_weights()[1].shape)

    #plot_data(history)
    #model.save(modelInitDir)
