import os

Train_Label_File = './data/split_train_test_64/train.txt'
Test_Label_File = './data/split_train_test_64/test.txt'
Train_Data_Path = './data/train_data/'
Test_Data_Path = './data/test_data/'

def rename():
    score = []
    ret_score = []
    with open(Train_Label_File, 'r') as train:
        for train_line in train:
            line_info = train_line[0:len(train_line)-1].split(' ')
            score.append(float(line_info[1]) * 2)
    with open(Test_Label_File, 'r') as test:
        for test_line in test:
            line_info = test_line[0:len(train_line)-1].split(' ')
            score.append(float(line_info[1]) * 2)
    max_score, min_score = max(score), min(score)
    print(max_score, min_score, sum(score) / len(score))

    #遍历文件重命名图片
    with open(Train_Label_File,'r') as train:
        for train_line in train:
            line_info = train_line[0:len(train_line)-1].split(' ')
            train_score = line_info[1]
            #train_score = ((train_score - min_score) / (max_score - min_score)) * 10
            #if train_score == 10.0:
            #    train_score = 9.9
            #ret_score.append(int(train_score))
            #train_score = str(int(train_score))
            original_name = Train_Data_Path + line_info[0]
            new_name = Train_Data_Path + train_score + '-' + line_info[0]
            if os.path.exists(original_name):
                os.rename(original_name,new_name)
                pass
            else:
                pass

    with open(Test_Label_File,'r') as test:
        for test_line in test:
            line_info = test_line[0:len(test_line) - 1].split(' ')
            test_score = line_info[1]
            #test_score = ((test_score - min_score) / (max_score - min_score)) * 10
            #if test_score == 10.0:
            #    test_score = 9.9
            #ret_score.append(int(test_score))
            #test_score = str(int(test_score))
            original_name = Test_Data_Path + line_info[0]
            new_name = Test_Data_Path + test_score + '-' + line_info[0]
            if os.path.exists(original_name):
                os.rename(original_name,new_name)
                pass
            else:
                pass
    #print(max(ret_score), min(ret_score), sum(ret_score) / len(ret_score))
    #print(len([score for score in ret_score if score == 1]))

def main():
    rename()

if __name__=='__main__':
    main()
