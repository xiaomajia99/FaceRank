import tensorflow as tf
import face_recognition
import numpy as np


from PIL import Image
from tensorflow.keras import models

#模型保存路径
MODEL_DIR = "./model/init/"
image_file_path = "./data/image_test/123456.jpg"
face_size = 128

#将图片转化成矩阵
def get_data(ImageFile):
    input = []
    #加载图片
    image = face_recognition.load_image_file(ImageFile)
    #获得图片中的面部集合
    face_locations = face_recognition.face_locations(image)
    #打印出原始图片中发现的面部数
    print("在原始图片中发现了 {} 张面部 .".format(len(face_locations)))
    # 图片中只有一张面部的情况
    if len(face_locations) == 1:
        # 获取面部在原始图片中的位置
        top, right, bottom, left = face_locations[0]
        # 打印出面部在图片中的位置
        print("面部所在位置的各像素点位置 Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
        # 截取面部图片
        face_image = image[top:bottom, left:right]
        # 将数组转化为图像
        pil_image = Image.fromarray(face_image)
        #重新设置图片大小
        resize_image = pil_image.resize((face_size,face_size))
        #将图片化为像素矩阵
        face = np.asarray(resize_image, dtype='float32') / 255.0
        face = np.reshape(face, [face_size, face_size, 3])
        input.append(face)
        data = np.array(input)
        data = np.reshape(data, [-1, face_size, face_size, 3])
        return data

    # 原始图片中存在多张面部的情况，遍历面部列表
    else:
        i = 0
        for face_location in face_locations:
            # 获取每一张面部在原始图片中的位置
            top, right, bottom, left = face_location
            # 打印出每一张面部在图片中的位置
            print("面部所在位置的各像素点位置 Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
            # 截取面部图片
            face_image = image[top:bottom, left:right]
            # 将数组转化为图像
            pil_image = Image.fromarray(face_image)
            # 重新设置图片大小
            resize_image = pil_image.resize((face_size, face_size))
            # 将图片化为像素矩阵
            face = np.asarray(resize_image, dtype='float32') / 255.0
            face = np.reshape(face, [face_size, face_size, 3])
            input.append(face)
            i = i + 1
        data = np.array(input)
        data = np.reshape(data, [-1, face_size, face_size, 3])
        return data

def predict(image_x, model_path):
    model = models.load_model(model_path)
    model.summary()
    y = model.predict(image_x)
    return y


if __name__ == '__main__':
    image_x = get_data(image_file_path)
    y_prd = predict(image_x, MODEL_DIR)
    #print(image_x)
    for y in y_prd:
        print(y)
        score = 0
        for i in range(0,len(y)):
            score += (i+1) * y[i]
        print(score)
