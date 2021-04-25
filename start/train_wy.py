import os.path
import shutil
import tensorflow.compat.v1 as tf
import data_wy
import numpy


from PIL import Image

#from tensorflow.contrib import slim
#from tensorflow.compat.v1.contrib import slim


tf.disable_eager_execution()

CKPT_MODEL_DIR = "./model/ckpt/"
CKPT_MODEL_NAME = "facerank.ckpt"
PB_MODEL_DIR = "./model/pb/"
PB_FILE_NAME = "facerank.pb"
TRAIN_DATA_DIR = "./data/resize_train_data/"

# 训练参数
batch_size = 30
learning_rate = 0.001
training_iters = 3000
STEPS=5001

# 网络结构
input_n = 64 * 64
classes_n = 10 
dropout = 0.75

x = tf.placeholder(tf.float32, [None, 64, 64, 3])
y = tf.placeholder(tf.float32, [None, classes_n])
keep_prob = tf.placeholder(tf.float32)

# 卷积参数
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 24])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 24, 96])),
    'wd1': tf.Variable(tf.random_normal([16*16*96, 1024])),
    'out': tf.Variable(tf.random_normal([1024, classes_n]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([24])),
    'bc2': tf.Variable(tf.random_normal([96])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([classes_n]))
}

def conv2d(x, w, b, step=1):
    """
    x是矩阵
    w是权重
    b是偏执
    step是步长
    """
    x = tf.nn.conv2d(x, w, strides = [1, step, step, 1], padding = 'SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    """
    最大值池化
    k是窗口大小
    """
    return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')



def conv_net(x, w, b, dropout):
    """
    ???
    """
    x = tf.reshape(x, shape = [-1, 64, 64, 3])
    conv1 = conv2d(x, w['wc1'], b['bc1'])
    conv1 = maxpool2d(conv1)

    conv2 = conv2d(conv1, w['wc2'], b['bc2'])
    conv2 = maxpool2d(conv2)

    fc1 = tf.reshape(conv2, [-1, w['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, w['wd1']), b['bd1'])
    fc1 = tf.nn.relu(fc1)

    fc1 = tf.nn.dropout(fc1, dropout)

    out = tf.add(tf.matmul(fc1, w['out']), b['out'])
    return out


def main():
    if not os.path.exists(CKPT_MODEL_DIR):  # 创建目录
        os.mkdir(CKPT_MODEL_DIR)
    if os.path.exists(PB_MODEL_DIR):  # 删除目录
        shutil.rmtree(PB_MODEL_DIR)
    if not os.path.exists(PB_MODEL_DIR):  # 创建目录
        os.mkdir(PB_MODEL_DIR)

    with tf.Session() as sess:
        #检测模型是否存在
        ckpt = tf.train.latest_checkpoint(CKPT_MODEL_DIR)
        #存在则断点续训
        if ckpt:
            pass
        else:
            #创建计算图，64为矩阵大小，10为分类数
            pred = conv_net(x, weights, biases, keep_prob)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
            optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
            
            
            correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            
            init = tf.global_variables_initializer()
            saver=tf.train.Saver()

            sess.run(init)

            imageList = os.listdir(TRAIN_DATA_DIR)
            imageList = [name for name in imageList if 'AF' in name]

            for batch_id in range(0, int(len(imageList) / batch_size) + 1):
                batch = imageList[batch_id*batch_size : batch_id*batch_size + batch_size]
                batch_xs = []
                batch_ys = []
                for imageName in batch:
                    id_tag = imageName.find("-")
                    score = imageName[0:id_tag]
                    img = Image.open(TRAIN_DATA_DIR + imageName)
                    batch_x = numpy.asarray(img, dtype='float32')
                    batch_x = numpy.reshape(batch_x, [64, 64, 3])
                    batch_xs.append(batch_x)

                    batch_y = numpy.asarray([0,0,0,0,0,0,0,0,0,0])
                    batch_y[int(score) - 1] = 1
                    batch_y = numpy.reshape(batch_y, [10,])
                    batch_ys.append(batch_y)
                batch_xs = numpy.asarray(batch_xs)
                batch_ys = numpy.asarray(batch_ys)
                if len(batch_xs) == 0:
                    continue
                print(batch_xs.shape, batch_ys.shape)
                print(batch_xs[0])
                print(batch_ys[0])
                exit()
                sess.run(optimizer, feed_dict={x:batch_xs, y:batch_ys, keep_prob:dropout})
                loss, acc = sess.run([cost, accuracy], feed_dict={x:batch_xs, y:batch_ys, keep_prob:1.0})
                print("Iter " + str(batch_id) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            if os.path.exists(CKPT_MODEL_DIR + CKPT_MODEL_NAME):
                os.remove(CKPT_MODEL_DIR + CKPT_MODEL_NAME)
            saver.save(sess, CKPT_MODEL_DIR + CKPT_MODEL_NAME)
            print('end', imageList[0:10])
            

if __name__ == '__main__':
    main()
