import os
import gc   # 内存回收模块
import cv2
import time
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from process import test_batch,after_process,root_path, \
                    task_list,recover,recover_softmax
from model import unet,get_input_output_ckpt
from util import tf_dice,tf_dice_index_norm,load_graph,get_newest,restore_from_pb,\
                 one_hot,readNiiAll,saveAsNiiGz,readImage,np_dice_index

test_path = os.path.join(root_path,"test",task_list[3])
test_list = os.listdir(test_path)

# hyper parameters
input_shape = (256,256)
num_class = 2
batch_size = 1
pattern = "pb"
ifout = True
ifprocess = False

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

saver = tf.train.import_meta_graph("ckpt/latest_model.meta")
meta_graph = tf.get_default_graph()
if pattern == "ckpt":
    # 下面这一大波操作其实都是在构建计算图而已，使用ckpt
    x,y_hat = get_input_output_ckpt(unet,input_shape,num_class)
    y = tf.placeholder(tf.float32,[None, *input_shape, num_class],name="input_y")
    y_softmax = tf.get_default_graph().get_tensor_by_name("softmax_y:0")
    y_result = tf.get_default_graph().get_tensor_by_name("segementation_result:0")

    # 这样构建计算图可以省去繁琐的原始网络结构定义，使用meta图
    graph = meta_graph

elif pattern == "pb":
    # 使用frozen_model
    graph = load_graph(get_newest("frozen_model"))

    x = graph.get_tensor_by_name("input_x:0")
    y = graph.get_tensor_by_name("input_y:0")
    y_softmax = graph.get_tensor_by_name("softmax_y:0")
    y_result = graph.get_tensor_by_name("segementation_result:0")
else:
    print("pattern must be ckpt or pb...")
    exit()

# dice这个ops scope在epoch 30之前没有所以epoch 30之前要自己定义，因为我找不到是哪一个
# epoch 30之后我做了name_scope的标记所以能够找到，epoch38之后学会了identity了,然而epoch55之前都没有在frozen_model中
# 保存dice这个节点，经过转换之后都有了
# 然而dice_index只能算softmax那种
dice_index = graph.get_tensor_by_name("dice:0")

if(ifout):
    test_result_root_path = "./test_result"
    if(not os.path.exists(test_result_root_path)):
        os.mkdir(test_result_root_path)
    # 不论是pb模式还是ckpt模式其实都是用的最新的那一套权重，然而pb和ckpt是一样的，就姑且用这个命名了
    addition_message = '_process' if ifprocess else '_raw'
    # os.path.split(get_newest("frozen_model"))[1]
    test_result_root_task_path = os.path.join(test_result_root_path,os.path.split(get_newest("frozen_model"))[1]+addition_message)
    if(not os.path.exists(test_result_root_task_path)):
        os.mkdir(test_result_root_task_path)
    out_txt = open("{}/result.txt".format(test_result_root_task_path),'w')

with graph.as_default():
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)
    print("sndaonsof)")
    if(pattern == "ckpt"):
        # 没有ckpt模型restore会失败，程序会退出，而之后也不会执行
        saver.restore(sess, "ckpt/latest_model")
    if(ifout):
        for one_patient in test_list:
            test_root_task_single_patient_path = os.path.join(test_result_root_task_path, one_patient)
            if(not os.path.exists(test_root_task_single_patient_path)):
                os.mkdir(test_root_task_single_patient_path)
            else:
                continue
            Spacing,Origin,one_patient_data = readNiiAll(os.path.join(test_path, one_patient,'data.nii'))
            _,_,one_patient_mask = readNiiAll(os.path.join(test_path, one_patient,'label.nii'))
            length = one_patient_data.shape[0]
            batch_object = test_batch(one_patient_data,one_patient_mask,num_class,input_shape)
            patient_mask_predict = []
            patient_soft_mask_predict = []
            while(1):
                batch,flag = batch_object.get_batch(batch_size)
                if(not flag):
                    break
                batch_test_x = batch[0]
                result,softmax = sess.run([y_result,y_softmax], feed_dict={x:batch_test_x})
                for j in range(batch_test_x.shape[0]):
                    patient_mask_predict.append(result[j])
                    patient_soft_mask_predict.append(softmax[j])
            del batch_object
            gc.collect()
            temp = recover(patient_mask_predict,one_patient_data.shape,ifprocess,num_class)
            real = one_hot(one_patient_mask,num_class)
            temp_sof = recover_softmax(patient_soft_mask_predict,one_patient_data.shape,num_class)
            print(temp_sof)
            dic = np_dice_index(temp_sof,real)
            dic_norm = np_dice_index(temp,real)
            print("patient{}:{} patient{}_softmax:{}".format(one_patient,dic_norm,one_patient,dic))
            out_txt.write("patient{}:{} patient{}_softmax:{}\n".format(one_patient,dic_norm,one_patient,dic))
            temp = np.argmax(temp,axis=-1)
            np.save(os.path.join(test_root_task_single_patient_path ,"softmax.npy"),temp_sof)
            saveAsNiiGz(temp, os.path.join(test_root_task_single_patient_path ,"test_label.nii.gz"), Spacing, Origin)
            del real
            del temp
            gc.collect()
        sess.close()
        out_txt.close()
    else:
        # several shot model
        for one_patient_data,one_patient_mask in block_fused:
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
                plt.subplot(131)
                plt.imshow(batch_test_x[0,:,:,0],cmap='gray')
                plt.title('data')
                plt.axis('off')
                plt.subplot(132)
                plt.imshow(np.argmax(batch_test_y[0],axis=-1),cmap='gray')
                plt.title('Groudtruth')
                plt.axis('off')
                result,dic,dic_norm = sess.run([y_result,dice_index,tf_dice_index_norm(y_softmax,y)], \
                                                feed_dict={x:batch_test_x,y:batch_test_y})
                plt.subplot(133)
                plt.imshow(result[0],cmap='gray')
                plt.axis('off')
                plt.title('test')
                plt.suptitle("test_dice_softmax:%.3f \ntest_dice_norm: %.3f \nnp_test_dice_norm: %.3f"%(dic,dic_norm,\
                             np_dice_index(one_hot(result,num_class),batch_test_y)))
                plt.show()
                for one_slice in result:
                    patient_mask_predict.append(one_slice)
            print(number)