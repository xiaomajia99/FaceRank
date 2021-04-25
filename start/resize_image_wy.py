from PIL import Image
import os

Train_Face_Path = './data/train_face/'
Test_Face_Path = './data/test_face/'
Resize_train_Path = './data/resize_train_data/'
Resize_test_Path = './data/resize_test_data/'

#重新设置图片尺寸
def resize_image(ImageFile,SaveFile):

    if os.path.exists(SaveFile):
        os.remove(SaveFile)
    img = Image.open(ImageFile)
    resize_image = img.resize((128,128))
    resize_image.save(SaveFile)

def main():
    train_images = os.listdir(Train_Face_Path)
    for image in train_images:
        ImageFile = Train_Face_Path+image
        SaveFile = Resize_train_Path+image
        resize_image(ImageFile,SaveFile)
    test_images = os.listdir(Test_Face_Path)
    for image in test_images:
        ImageFile = Test_Face_Path+image
        SaveFile = Resize_test_Path+image
        resize_image(ImageFile,SaveFile)

if __name__=='__main__':
    main()