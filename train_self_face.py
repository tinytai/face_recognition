# -*- coding: utf-8 -*-
'''
    made by hc
    2016-10-7
'''
import cv
import tensorflow as tf
import numpy as np
import Image
import random
import matplotlib.pyplot as plt
from numpy.random import randn

# 待输入的占位符,x为图像数据的维度，y为分类数据的维度
# 留一个Question 以数组进行的分类目测可行了，和以数字进行的分类训练呢？why?
x = tf.placeholder("float", shape=[None, 164, 164, 3])
y_ = tf.placeholder("float", shape=[None, 4])
keep_prob = tf.placeholder("float")

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def read_train_model():
    #载入data
    arrTemp=randn(4)
    arrGonglin=randn(150,164,164,3)
    arrQinxuebin=randn(150,164,164,3)
    arrSunming=randn(150,164,164,3)
    arrLvyuanjie=randn(150,164,164,3)
    
    arrGonglin=np.asarray(arrGonglin,dtype=np.float32)
    arrQinxuebin=np.asarray(arrQinxuebin,dtype=np.float32)
    arrSunming=np.asarray(arrSunming,dtype=np.float32)
    arrLvyuanjie=np.asarray(arrLvyuanjie,dtype=np.float32)
    for i in range(150):
        #164x164
        print i
        imgGonglin=cv.LoadImage("/home/deeplearn/Desktop/qin/gonglin/"+str(i)+".jpg",1)
        imgQinxuebin=cv.LoadImage("/home/deeplearn/Desktop/qin/qinxuebin/"+str(i)+".jpg",1)
        imgSunming=cv.LoadImage("/home/deeplearn/Desktop/qin/sunming/"+str(i)+".jpg",1)
        imgLvyuanjie=cv.LoadImage("/home/deeplearn/Desktop/qin/sunming/"+str(i)+".jpg",1)
        for h in range(164):
            for w in range(164):
                pixPair=cv.Get2D(imgGonglin,h,w)
                arrTemp=np.array(pixPair)
                arrGonglin[i,w,h,0]=arrTemp[0]
                arrGonglin[i,w,h,1]=arrTemp[1]
                arrGonglin[i,w,h,2]=arrTemp[2]

                pixHorse=cv.Get2D(imgQinxuebin,h,w)
                arrTemp=np.array(pixHorse)
                arrQinxuebin[i,w,h,0]=arrTemp[0]
                arrQinxuebin[i,w,h,1]=arrTemp[1]
                arrQinxuebin[i,w,h,2]=arrTemp[2]

                pixCat=cv.Get2D(imgSunming,h,w)
                arrTemp=np.array(pixCat)
                arrSunming[i,w,h,0]=arrTemp[0]
                arrSunming[i,w,h,1]=arrTemp[1]
                arrSunming[i,w,h,2]=arrTemp[2]

                pixCat=cv.Get2D(imgLvyuanjie,h,w)
                arrTemp=np.array(pixCat)
                arrLvyuanjie[i,w,h,0]=arrTemp[0]
                arrLvyuanjie[i,w,h,1]=arrTemp[1]
                arrLvyuanjie[i,w,h,2]=arrTemp[2]

        #cv.Zero(imgPair)
        #cv.Zero(imgHorse)
        #cv.Zero(imgCat)
    print "Load data finished!"

    zeroArray = np.zeros((4,4))
    for nI in range(0,4):
        zeroArray[nI][nI] = 1

    print zeroArray
    return arrGonglin,arrQinxuebin,arrSunming,arrLvyuanjie,zeroArray


# Network Parameters
n_input = 164 * 164 * 4# 输入数据的维数
n_classes = 4         # 标签维度
dropout = 0.75         # Dropout, probability to keep units

 
def model():
    W_conv1 = weight_variable([3, 3, 3, 16])     
    b_conv1 = bias_variable([16])

    W_conv2 = weight_variable([2, 2, 16, 32])            
    b_conv2 = bias_variable([32])

    W_conv3 = weight_variable([3, 3, 32, 64])             
    b_conv3 = bias_variable([64])

    W_conv4 = weight_variable([2, 2, 64, 128])            
    b_conv4 = bias_variable([128])

    W_conv5 = weight_variable([2, 2, 128, 256])            
    b_conv5 = bias_variable([256])

    W_fc1 = weight_variable([4 * 4 * 256, 256])  
    b_fc1 = bias_variable([256])

    W_fc2 = weight_variable([256, 256])
    b_fc2 = bias_variable([256])

    W_fc3 = weight_variable([256, 4])
    b_fc3 = bias_variable([4])
    
    print(1)
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    print(2)
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    h_pool4 = max_pool_2x2(h_conv4)

    h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
    h_pool5 = max_pool_2x2(h_conv5)
    
    print(3)
    h_pool4_flat = tf.reshape(h_pool5, [-1, 4 * 4 * 256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
    print(4)

    print(5)
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
    print(6)

    y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
    print "tf.run(y_conv)"
    rmse = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    return y_conv, rmse

def save_model(saver,sess,save_path):
    path = saver.save(sess, save_path)
    print 'model save in :{0}'.format(path)

def readImage2Array(path):
    im = Image.open(path)
    ims = im.resize((164,164))
    ims = ims.convert("L")
    out=np.asarray(ims)
    out=np.float32(out/255.0)
    print(out.shape)
    out=out.reshape((1,164,164,1)) 
    return out

def readJpg2Array(path):
    arrTemp=randn(4)
    arrPic=randn(1,164,164,3)
    arrPic=np.asarray(arrPic,dtype=np.float32)
    imgPair=cv.LoadImage(path,1)

    for h in range(164):
        for w in range(164):
            pixPair=cv.Get2D(imgPair,h,w)
            arrTemp=np.array(pixPair)
            arrPic[0,w,h,0]=arrTemp[0]
            arrPic[0,w,h,1]=arrTemp[1]
            arrPic[0,w,h,2]=arrTemp[2]
    return arrPic

def readCarConfig(path):
    fo=open(path,'r')
    strs=fo.read(-1)
    array = strs.split('\r\n')
    resultArr = []
    for nIndex in range(4):
        tempArr=array[nIndex].split(' ')
        tempstr=tempArr[1]
        f=tempstr.decode("gb2312")
        resultArr.append(f)
    fo.close()
    return resultArr


if __name__ == '__main__':
    sess         = tf.InteractiveSession()
    y_conv, rmse = model()
    train_step   = tf.train.AdamOptimizer(1e-3).minimize(rmse)
    sess.run(tf.initialize_all_variables())

    current_epoch = 0
    train_index = range(4)
    #random.shuffle(train_index)

    X1_train,X2_train,X3_train,X4_train,y_train=read_train_model()
    print "loading ok... start training"

    saver = tf.train.Saver()
    best_validation_loss=1.0
    print 'begin training..., train dataset size:{0}'.format(1521)

    temp_X_train=randn(4,164,164,3)
    print ("please input 1 for traning 2 for test!")
    flag = raw_input(">")

if flag == "1":  
    for i in xrange(1000):   
        for k in xrange(150):
            for m in xrange(3):
                temp_X_train[0,:,:,m]= X1_train[k,:,:,m]#gonglin
                temp_X_train[1,:,:,m]= X2_train[k,:,:,m]#qinxuebin
                temp_X_train[2,:,:,m]= X3_train[k,:,:,m]#sunming
                temp_X_train[3,:,:,m]= X4_train[k,:,:,m]#lvyuanjie
            #print 'training ...k' + str(k)
            train_step.run(feed_dict={x:temp_X_train[train_index],y_:y_train[train_index], keep_prob:0.95})
            train_loss = rmse.eval(feed_dict={x:temp_X_train, y_:y_train, keep_prob: 0.95})
        print 'epoch {0} done! validation loss:{1}'.format(i, train_loss*100.0)
        print 'training ...' + str(i)

    save_path = saver.save(sess, "self_face.ckpt")
else:
    saver.restore(sess, "self_face.ckpt")
    for i in xrange(150):
        out = readJpg2Array("/home/deeplearn/Desktop/qin/qinxuebin/"+str(i)+".jpg")
        y_batch = y_conv.eval(feed_dict={x:out, keep_prob:1.0})
        carConfig=readCarConfig("/home/deeplearn/Desktop/qin/resultFace.txt")
        result = carConfig[y_batch[0].argmax()]
        print str(i)+":" + result

    print """"""""
    for i in xrange(150):
        out = readJpg2Array("/home/deeplearn/Desktop/qin/lvyuanjie/"+str(i)+".jpg")
        y_batch = y_conv.eval(feed_dict={x:out, keep_prob:1.0})
        carConfig=readCarConfig("/home/deeplearn/Desktop/qin/resultFace.txt")
        result = carConfig[y_batch[0].argmax()]
        print str(i)+":" + result

    print """"""""
    for i in xrange(150):
        out = readJpg2Array("/home/deeplearn/Desktop/qin/sunming/"+str(i)+".jpg")
        y_batch = y_conv.eval(feed_dict={x:out, keep_prob:1.0})
        carConfig=readCarConfig("/home/deeplearn/Desktop/qin/resultFace.txt")
        result = carConfig[y_batch[0].argmax()]
        print str(i)+":" + result

    print """"""""
    for i in xrange(150):
        out = readJpg2Array("/home/deeplearn/Desktop/qin/gonglin/"+str(i)+".jpg")
        y_batch = y_conv.eval(feed_dict={x:out, keep_prob:1.0})
        carConfig=readCarConfig("/home/deeplearn/Desktop/qin/resultFace.txt")
        result = carConfig[y_batch[0].argmax()]
        print str(i)+":" + result        

    #out = readJpg2Array("/home/deeplearn/Desktop/qin/color/ma1.jpg")
    #y_batch = y_conv.eval(feed_dict={x:out, keep_prob:1.0})
    #carConfig=readCarConfig("/home/deeplearn/Desktop/qin/result.txt")
    #result = carConfig[y_batch[0].argmax()]
    #print result




