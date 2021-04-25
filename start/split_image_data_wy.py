import os
import shutil

Image_Path = './data/Images/'
Image_Output_Path = './data/'
Train_Label_File = './data/split_train_test_64/train.txt'
Test_Label_File = './data/split_train_test_64/test.txt'
Train_Data_Path = Image_Output_Path + '/train_data/'
Test_Data_Path = Image_Output_Path + '/test_data/'

def split(Train_Data_Path,Test_Data_Path):
    with open(Train_Label_File,'r') as train:
        for train_line in train:
            line_info = train_line[0:len(train_line)-1].split(' ')
            shutil.copyfile(Image_Path + line_info[0], Train_Data_Path + line_info[1] + "-" + line_info[0])
            #shutil.copyfile(Image_Path + train_name, Train_Data_Path + train_name)
            #os.rename(Image_Path+train_name,Train_Data_Path+train_name)
    with open(Test_Label_File,'r') as test:
        for test_line in test:
            line_info = test_line[0:len(test_line)-1].split(' ')
            shutil.copyfile(Image_Path + line_info[0], Test_Data_Path + line_info[1] + "-" + line_info[0])
            #os.rename(Image_Path+test_name,Test_Data_Path+test_name)
            #shutil.copyfile(Image_Path + test_name, Test_Data_Path + test_name)

def main():
    split(Train_Data_Path,Test_Data_Path)

if __name__ == '__main__':
    main()
