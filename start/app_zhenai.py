import tensorflow as tf
import face_recognition
import numpy as np
import os
import shutil


from PIL import Image
from tensorflow.keras import models

#模型保存路径
MODEL_DIR = "./model/init/"
image_file_path = "./data/zhenai/image/"
new_file_path = "./data/zhenai/score/"
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

def predict(image_x, model):
    y = model.predict(image_x)
    return y

def get_model(model_path):
    return models.load_model(model_path)

def get_file(dir_path):
    file_names = os.listdir(dir_path)
    return file_names

if __name__ == '__main__':
    file_names = get_file(image_file_path)
    model = get_model(MODEL_DIR)
    score = []
    print(file_names[0:5])
    #file_names = file_names[0:5]
    for file_name in file_names:
        image_x = get_data(image_file_path + file_name)
        if (len(image_x) < 1):
            continue
        y_prd = predict(image_x, model)
        y_prd = y_prd[0]
        print(round(y_prd[0], 3), file_name)
        score.append(round(y_prd[0], 3))
        shutil.copyfile(image_file_path + file_name,  new_file_path + str(round(y_prd[0], 3)) + "_" + file_name)
    print(max(score), min(score), len(score), round(np.mean(score), 3), round(np.median(score),3))
    
