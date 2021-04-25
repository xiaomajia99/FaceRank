
import os
import urllib
import requests
import face_recognition

file_path = "./data/zhenai/filepath.csv"

def downFile(file_path):
    with open(file_path, 'r') as data_info:
        for line in data_info:
            lines = line[0:len(line)-1].split(',')
            r = requests.get(lines[3])
            print(type(r.content))
            print(r.content)
            #file_name = lines[0]
            image = face_recognition.load_image_file(lines[3])
            print(image)

            #with open('./data/zhenai/image/' + file_name + '.jpg', 'wb') as img:
            #    img.write(r.content)
            break
def load_image(ImageFile):
    image = face_recognition.load_image_file(ImageFile)
    print(type(image))
    print(image)

if __name__ == "__main__":

    downFile(file_path)
    load_image("./data/zhenai/image/111514600.jpg")
    print("hello world")