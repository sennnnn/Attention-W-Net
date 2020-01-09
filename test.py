import os
import time
import random
import preprocess
import tensorflow as tf
import numpy as np
from preprocess import test_batch,read_test_data
from model import unet,get_input_output_ckpt
from util import dice
import matplotlib.pyplot as plt

root_path = preprocess.root_path
task_list = preprocess.task_list
test_path = os.path.join(root_path,"test",task_list[2])
test_list = os.listdir(test_path)

# hyper parameters
input_shape = (256,256)
num_class = 7
batch_size = 4

start = time.time()
data,mask = read_test_data(test_path,test_list,input_shape)
end = time.time()
print("spend time:%.2fs\ndata_shape:{} mask_shape:{}".format(data.shape,mask.shape)%(end-start))

# batch_object = test_batch()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

# # 下面这一大波操作其实都是在构建计算图而已
# x,y_hat = get_input_output_ckpt(unet,input_shape,num_class)
# y = tf.placeholder(tf.float32,[None, *input_shape, num_class],name="label")
# y_softmax = tf.get_default_graph().get_tensor_by_name("softmax_y:0")
# y_result = tf.get_default_graph().get_tensor_by_name("segementation_result:0")

# 这样构建计算图可以省去繁琐的原始网络结构定义
saver = tf.train.import_meta_graph("ckpt/latest_model.meta")
x = tf.get_default_graph().get_tensor_by_name("input:0")
y = tf.get_default_graph().get_tensor_by_name("label:0")
y_softmax = tf.get_default_graph().get_tensor_by_name("softmax_y:0")
y_result = tf.get_default_graph().get_tensor_by_name("segementation_result:0")

# dice这个ops scope在epoch 30之前没有所以epoch 30之前要自己定义，因为我找不到是哪一个
# epoch 30之后我做了name_scope的标记所以能够找到
dice_index = dice(y_softmax,y)

saver = tf.train.Saver()
init = tf.global_variables_initializer()
with tf.Session(config=config) as sess:
    sess.run(init)
    # 没有ckpt模型restore会失败，程序会退出，而之后也不会执行
    saver.restore(sess, "ckpt/latest_model")
    
    for one_patient_data,one_patient_mask in zip(data,mask):
        # batch对象只能走一个轮回就结束了，单个病人不能循环
        batch_object = test_batch(one_patient_data,one_patient_mask,num_class)
        number = 0
        patient_mask_predict = []
        while(1):
            batch,flag = batch_object.get_batch(batch_size)
            if(not flag):
                break
            batch_test_x,batch_test_y = batch[0],batch[1]
            number += batch_test_x.shape[0]
            plt.subplot(221)
            plt.imshow(batch_test_x[0,:,:,0],cmap='gray')
            plt.axis('off')
            plt.subplot(222)
            plt.imshow(np.argmax(batch_test_y[0],axis=-1),cmap='gray')
            plt.axis('off')
            result,dic = sess.run([y_result,dice_index], feed_dict={x:batch_test_x,y:batch_test_y})
            plt.subplot(223)
            plt.imshow(batch_test_x[0,:,:,0],cmap='gray')
            plt.axis('off')
            plt.subplot(224)
            plt.imshow(result[0],cmap='gray')
            plt.axis('off')
            plt.title("dice:%.3f"%(dic))
            plt.show()
            for one_slice in result:
                patient_mask_predict.append(one_slice)
        print(number)

