import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from process import root_path,task_list,read_train_data
from util import one_hot,weight_loss

def prior1(ifpixel):
    train_path = os.path.join(root_path,"train",task_list[3])
    train_list = os.listdir(train_path)

    input_shape = (256,256)
    num_class = 2
    index = 10

    start = time.time()
    data,mask = read_train_data(train_path,train_list,input_shape)
    end = time.time()
    print("read ALL...\nspend time:{}\ndata shape:{} mask shape:{}".format(end-start,data.shape,mask.shape))

    positive_data = []
    positive_mask = []
    negetive_data = []
    negetive_mask = []
    pixel=[0]*2
    flag = 0
    for i,j in zip(data,mask):
        if(ifpixel):
            for w in j:
                for h in w:
                    pixel[h] += 1
        if(np.max(j) == 0):
            negetive_data.append(i)
            negetive_mask.append(j)
        else:
            positive_data.append(i)
            positive_mask.append(j)
        flag += 1
        if(flag % 100 == 0):
            print("There are {} slice down...".format(flag))
    print(pixel)
    print("positive:{}".format(len(positive_data)),"negetive:{}".format(len(negetive_data)))

def prior2():
    bg_tumor = [317809563, 236645]
    rate_bg = bg_tumor[1]/(bg_tumor[0]+bg_tumor[1]);rate_tumor = bg_tumor[0]/(bg_tumor[0]+bg_tumor[1])
    print(rate_bg,rate_tumor)

def prior3():
    norm1 = tf.random_uniform([1,2])
    va1 = tf.Variable(norm1)
    norm2 = tf.random_uniform([1,2])
    va2 = tf.Variable(norm2)
    # weight = [0.0007440585488760174,0.999255941451124]
    weight = tf.constant([1.,1.])
    loss = tf.reduce_mean(weight_loss(va1,va2,weight))
    loss_ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=va1,logits=va2))
    # print(loss)
    # print(loss_)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run([va1,tf.expand_dims(va1,axis=-1)]))
        print(sess.run([loss,loss_]))
prior1(False)
prior3()