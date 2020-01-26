import os
import gc   # 内存回收模块
import cv2
import time
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from process import test_batch,after_process, \
                    config_dict,recover
from util import tf_dice,tf_dice_index_norm,load_graph,get_newest,restore_from_pb,\
                 one_hot,readNiiAll,saveAsNiiGz,readImage,np_dice_index,test_result_dir_initial,\
                 tuple_string_to_tuple

OAR_OR_GTV = 'OAR' # OAR:task1 or task3 GTV:task2 or task4
task_number = 'task3'
config_dict_lite = config_dict[OAR_OR_GTV][task_number]
root_path = config_dict['root_path']
task_name = config_dict_lite['task_name']
test_path = os.path.join(root_path,"test",task_name)
test_list = os.listdir(test_path)

# hyper parameters
input_shape = tuple_string_to_tuple(config_dict_lite['input_shape'])
num_class = config_dict_lite['obj_num']+1
batch_size = 2
ifprocess = False
crop_x_range = tuple_string_to_tuple(config_dict_lite['crop_x_range'])
crop_y_range = tuple_string_to_tuple(config_dict_lite['crop_y_range'])
model_select = "unet" # "unet" or "unet-CAB"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

graph = load_graph(get_newest("frozen_model/{}/{}".format(task_name,model_select)))
x = graph.get_tensor_by_name("input_x:0")
y = graph.get_tensor_by_name("input_y:0")
y_result = graph.get_tensor_by_name("segementation_result:0")

test_result_root_task_path,out_txt = test_result_dir_initial(ifprocess,task_name,model_select)

with graph.as_default():
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)
    all_result = [0]*num_class
    for one_patient in test_list:
        test_root_task_single_patient_path = os.path.join(test_result_root_task_path, one_patient)
        if(not os.path.exists(test_root_task_single_patient_path)):
            os.mkdir(test_root_task_single_patient_path)
        else:
            continue
        Spacing,Origin,one_patient_data = readNiiAll(os.path.join(test_path, one_patient,'data.nii'))
        _,_,one_patient_mask = readNiiAll(os.path.join(test_path, one_patient,'label.nii'))
        length = one_patient_data.shape[0]
        batch_object = test_batch(one_patient_data,one_patient_mask,num_class,input_shape,crop_x_range,crop_y_range)
        patient_mask_predict = []
        while(1):
            batch,flag = batch_object.get_batch(batch_size)
            if(not flag):
                break
            batch_test_x = batch[0]
            result = sess.run(y_result, feed_dict={x:batch_test_x})
            for j in range(batch_test_x.shape[0]):
                patient_mask_predict.append(result[j])
        temp = recover(patient_mask_predict,one_patient_data.shape,ifprocess,num_class,crop_x_range,crop_y_range,task_name)
        real = one_hot(one_patient_mask,num_class)
        dic_norm = np_dice_index(temp,real)
        all_result[0] += dic_norm
        dice_string = []
        for channel in range(1,real.shape[-1]):
            channel_wise_dice = np_dice_index(temp[...,channel],real[...,channel])
            all_result[channel] += channel_wise_dice
            dice_string.append("Object{}:%.3f ".format(channel)%(channel_wise_dice))
        dice_string = "".join(dice_string)
        print("patient{}:\ndice:%.3f\n{}".format(one_patient,dice_string)%(dic_norm))
        out_txt.write("patient{}:\ndice:%.3f\n{} \n".format(one_patient,dice_string)%(dic_norm))
        temp = np.argmax(temp,axis=-1)
        saveAsNiiGz(temp, os.path.join(test_root_task_single_patient_path ,"test_label.nii.gz"), Spacing, Origin)
    all_result = [x/len(test_list) for x in all_result]
    dice_string = []
    dice_string.append("Background:%.3f "%(all_result[0]))
    [dice_string.append("Object{}:%.3f ".format(i)%(all_result[i])) for i in range(1,num_class)]
    print("".join(dice_string))
    out_txt.write("".join(dice_string)+'\n')
    sess.close()
    out_txt.close()