import os
import cv2
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from util import one_hot,readImage,get_config_dict

config_dict = get_config_dict("task.yaml")

def _preprocess_oneslice(one_slice,tran,filp,angle):
    """
    Cstro Challenge:The shape of slice is mainly 512x512.
    Project:flip、rotate、standardization
    args:
    one_slice : A numpy mat and it shapes like 2x512x512 and the data type is [np.float32,np.uint8]
    
    return:
    tuple:(data_slice,mask_slice)
    """
    out_data = one_slice[0]
    out_mask = one_slice[1]
    h,w = out_data.shape[:2]
    out_data = out_data.astype(np.float32)
    out_data = out_data + abs(np.min(out_data))
    # 获得旋转矩阵
    if(angle!=0):
        center = (w//2,h//2)
        random_angle = random.randint(-1*angle,angle)
        rotate_matrix = cv2.getRotationMatrix2D(center,random_angle,1)
        # 旋转
        out_data = cv2.warpAffine(out_data, rotate_matrix, (w,h))
        out_mask = cv2.warpAffine(out_mask, rotate_matrix, (w,h))
    if(tran==True):
        x_shift = random.randint(-10,10)
        y_shift = random.randint(-10,10)
        # 构建平移矩阵
        trans_matrix = np.array([[1,0,x_shift],[0,1,y_shift]],dtype=np.float32)
        out_data = cv2.warpAffine(out_data, trans_matrix, (w,h))
        out_mask = cv2.warpAffine(out_mask, trans_matrix, (w,h))

    if(filp==True):
        random_state = random.randint(-6,2)
        if(random_state>=0):
            out_data = cv2.flip(out_data, random_state-1)
            out_mask = cv2.flip(out_mask, random_state-1)
        # 水平翻转
        # out = cv2.flip(out,1)
        # 垂直翻转
        # out = cv2.flip(out,0)
        # 水平垂直翻转
        # out = cv2.flip(out,-1)
    
    # 最大最小归一化、函数转化归一化、z-score标准化，仅针对数据而不是标签
    """
    z-score Standardization
    """
    mean,stdDev = cv2.meanStdDev(out_data)
    mean,stdDev = mean[0][0],stdDev[0][0]
    out_data = (out_data-mean)/stdDev
    return (out_data,out_mask)

def length_norm_crop(slice,general_shape,crop_x_range,crop_y_range):
    # Lung
    # crop_x_range:(117,417) crop_y_range:(99,399)
    # nasopharynx
    # crop_x_range:(160,384) crop_y_range:(99,399)
    slice = cv2.resize(slice, general_shape)
    slice = slice[np.ix_(range(*crop_x_range),range(*crop_y_range))]
    return slice

def read_train_data(train_path,train_list,input_shape,crop_x_range,crop_y_range):
    """
    Read all of the training data.
    Args:
    train_path:the training data root path.
    train_list:the training data list.
    input_shape:the shape of the net input.

    Return:
    data:the training data block.
    mask:the training label block.
    """
    data_list = []
    mask_list = []
    for patient in train_list:
        data = readImage(os.path.join(train_path,patient,'data.nii'))
        mask = readImage(os.path.join(train_path,patient,'label.nii'))
        for i,j in zip(data,mask):
            i = length_norm_crop(i,(512,512),crop_x_range,crop_y_range)
            j = length_norm_crop(j,(512,512),crop_x_range,crop_y_range)
            i = cv2.resize(i.astype(np.float32),input_shape)
            j = cv2.resize(j,input_shape)
            data_list.append(i)
            mask_list.append(j)
    return data_list,mask_list

def swap(a,b):
    temp = a
    a = b
    b = temp
    return a,b

def OAR_epoch_read(data,mask,rate):
    data_list = []
    mask_list = []
    for i,j in zip(data,mask):
        temp = random.randint(1,rate[1])
        if(temp <= rate[0]):
            data_list.append(i)
            mask_list.append(j)
    return np.array(data_list),np.array(mask_list)

def GTV_epoch_read(data,mask,positive_get_rate,negetive_get_rate):
    # 鼻咽癌的GTV中正负样本比例为5411:740
    data_list = []
    mask_list = []
    # 正样本中提取一部分，负样本中提取一部分
    for i,j in zip(data,mask):
        if(np.max(j) == 0):
            flag = random.randint(1,negetive_get_rate[1])
            if(flag <= negetive_get_rate[0]):
                data_list.append(i)
                mask_list.append(j)
        else:
            flag = random.randint(1,positive_get_rate[1])
            if(flag <= positive_get_rate[0]):
                data_list.append(i)
                mask_list.append(j)
    return np.array(data_list),np.array(mask_list)

def epoch_train_get(data_list,mask_list,folder_rate):
    """
    单个epoch只读入训练集的部分，这是因为我感觉训练集对于网络来说太大了，反馈太慢，训练耗时太长
    """
    total_length = len(data_list)
    length = int(folder_rate*total_length)
    for i in range(total_length):
        temp = random.randint(0,i)
        data_list[i],data_list[temp] = swap(data_list[i],data_list[temp])
        mask_list[i],mask_list[temp] = swap(mask_list[i],mask_list[temp])
    return np.array(data_list[0:length]),np.array(mask_list[0:length])

class train_batch(object):
    def __init__(self,data_block,mask_block,trans,flip,angle,num_class):
        self.data = data_block
        self.mask = mask_block
        self.flip = flip
        self.angle = angle
        self.num_class = num_class
        self.length = data_block.shape[0]
        self.trans = trans

    def get_batch(self,batch_size):
        length = self.length
        index  = random.randint(0,length-batch_size)
        batch_data = []
        batch_mask = []
        for i in range(index,index+batch_size):
            temp_slice_data,temp_slice_mask = _preprocess_oneslice((self.data[i],\
                            self.mask[i]),self.trans,self.flip,self.angle)
            batch_data.append(temp_slice_data)
            batch_mask.append(temp_slice_mask)
        return np.expand_dims(np.array(batch_data),-1),one_hot(np.array(batch_mask),self.num_class)

class test_batch(object):
    def __init__(self,one_patient_data,one_patient_mask,num_class,input_shape,crop_x_range,crop_y_range):
        self.data = one_patient_data
        self.mask = one_patient_mask
        self.num_class = num_class
        self.length = one_patient_data.shape[0]
        self.index = 0
        self.input_shape = input_shape
        self.x_range = crop_x_range
        self.y_range = crop_y_range

    def __do(self,data,mask,start,end):
        batch_data = []
        batch_mask = []
        data = data[start:end]
        mask = mask[start:end]
        for i,j in zip(data,mask):
            i = length_norm_crop(i.astype(np.float32),(512,512),self.x_range,self.y_range)
            j = length_norm_crop(j.astype(np.uint8),(512,512),self.x_range,self.y_range)
            i = cv2.resize(i.astype(np.float32),self.input_shape)
            j = cv2.resize(j,self.input_shape)
            i,j = _preprocess_oneslice((i,j),False,False,0)
            batch_data.append(i)
            batch_mask.append(j)
        batch_data = np.expand_dims(np.array(batch_data),axis=-1)
        batch_mask = one_hot(np.array(batch_mask),self.num_class)
        return batch_data,batch_mask

    def get_batch(self,batch_size):
        # 当已经把一个病人遍历完了的时候那么就会输出空
        if(self.index >= self.length):
            return [],False

        if(self.index + batch_size > self.length):
            start = self.index
            end = self.length
            self.index += batch_size
            return self.__do(self.data,self.mask,start,end),True
        else:
            start = self.index
            end = self.index + batch_size
            self.index += batch_size
            return self.__do(self.data,self.mask,start,end),True

def recover(patient_mask_predict,shape,ifprocess,num_class,crop_x_range,crop_y_range,task_name):
    length = len(patient_mask_predict)
    w = crop_x_range[1]-crop_x_range[0]
    h = crop_y_range[1]-crop_y_range[0]
    out = np.array([cv2.resize(x.astype(np.uint8), (w,h)) for x in patient_mask_predict],dtype=np.uint8)
    temp = np.zeros(shape=shape, dtype=np.uint8)
    for i in range(length):
        temp_slice = np.zeros(shape=(512,512),dtype=np.uint8)
        temp_slice[np.ix_(range(*crop_x_range),range(*crop_y_range))] = out[i]
        temp[i] = cv2.resize(temp_slice, tuple(temp[i].shape))
    # After postprocessing...
    temp = one_hot(temp, num_class)
    print(temp.shape)
    if(ifprocess):
        temp = np.array([after_process(temp[i],task_name) for i in range(temp.shape[0])])
    return temp

def dilate(src, kernel_size=(3,3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    dst = cv2.dilate(src,kernel,iterations=2)
    return dst

def erode(src,kernel_size=(3,3)):
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,kernel_size)
    dst = cv2.erode(src,kernel,iterations=1)
    return dst

def left_lung_after_process(src,benchmark):
    # 输入的是一张slice，src为左肺切片，benchmark为右肺切片
    # 即将右肺预测到左肺的部分给去掉
    begin = 0
    end = 0
    # 首先获得搜索开始的y轴下标
    for i in range(src.shape[1]):
        if(np.max(src[:,i]) != 0):
            begin = i
            break

    for j in range(begin,src.shape[1]):
        if(np.max(src[:,j]) == 0):
            end = j
            break

    for k in range(begin,end):
        for x_index in range(src.shape[0]):
            if(benchmark[x_index,k] != 0):
                src[x_index,k] = 2

    return src
    
def right_lung_after_process(src,benchmark):
    # 输入的是一张slice，src为右肺切片，benchmark为左肺切片
    # 即将左肺预测到右肺的部分给去掉
    begin = 0
    end = 0
    # 首先获得搜索开始的y轴下标 
    for i in range(src.shape[1])[::-1]:
        if(np.max(src[:,i]) != 0):
            begin = i
            break

    for j in range(0,begin)[::-1]:
        if(np.max(src[:,j]) == 0):
            end = j
            break

    for k in range(end,begin)[::-1]:
        for x_index in range(src.shape[0]):
            if(benchmark[x_index,k] != 0):
                src[x_index,k] = 2
    return src

def findMaxContours(src):
    # 输入二值化图像，查找图像轮廓最大面积，当比较小的时候就不做一些处理
    contours, _ = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    area = []
    if(len(contours) == 0):
        return 0
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    return area[np.argmax(np.array(area))]

def after_process(one_slice,task):
    # after one_hot multi slice block
    # 这里只是用到了简单的形态学腐蚀，那么我们再假设，左肺最右点不可能有右肺，右肺最左点不可能有左肺
    # 搜索左肺最右点，即y最大点
    # 输入进来的为病人的完整block，且已经one_hot编码
    one_slice_mid_dst = np.expand_dims(one_slice[...,0],axis=-1)
    # 腐蚀处理
    for j in range(1,one_slice.shape[-1]):
        temp = np.expand_dims(erode(one_slice[...,j]),axis=-1)
        one_slice_mid_dst = np.concatenate([one_slice_mid_dst,temp],axis=-1)
    # 左右肺排查，已经膨胀处理，这是针对肺部的操作
    # 测试出来的有效面积
    if(task == "Thoracic_OAR"):
        if(findMaxContours(one_slice_mid_dst[...,2]) > 800):
            one_slice_mid_dst[...,2] = left_lung_after_process(one_slice_mid_dst[...,2],one_slice_mid_dst[...,1])
        if(findMaxContours(one_slice_mid_dst[...,1]) > 800):
            one_slice_mid_dst[...,1] = right_lung_after_process(one_slice_mid_dst[...,1],one_slice_mid_dst[...,2])
    
    one_slice_dst = np.expand_dims(one_slice_mid_dst[...,0],axis=-1)
    for j in range(1,one_slice.shape[-1]):
        temp = np.expand_dims(dilate(one_slice_mid_dst[...,j]),axis=-1)
        one_slice_dst = np.concatenate([one_slice_dst,temp],axis=-1)
    return one_slice_dst

if __name__ == "__main__":
    # train batch get example
    train_path = os.path.join(root_path,"train",task_list[2])
    train_list = os.listdir(train_path)
    start = time.time()
    data,mask = read_train_data(train_path,train_list)
    end = time.time()
    data_train,data_valid,mask_train,mask_valid = train_test_split(data,mask,test_size=0.1,shuffle=True)
    print("spend time:%.2fs\ndata_shape:{} mask_shape:{}".format(data.shape,mask.shape)%(end-start))
    train_batch_object,valid_batch_object = train_batch(data_train,mask_train,True,15,7),train_batch(data_valid,mask_valid,True,15,7)
    # begin show
    while(1):
        batch_train_x,batch_train_y = train_batch_object.get_batch(1)
        batch_valid_x,batch_valid_y = valid_batch_object.get_batch(1)
        print(batch_train_x.shape," ",batch_train_y.shape)
        print(batch_valid_x.shape," ",batch_valid_y.shape)
        for i,j,l,m in zip(batch_train_x,batch_train_y,batch_valid_x,batch_valid_y):
            i = i[...,0]
            j = np.argmax(j,axis=-1)
            l = l[...,0]
            m = np.argmax(m,axis=-1)
            plt.subplot(221)
            plt.imshow(i,cmap='gray')
            plt.title("{}x{}".format(*tuple(i.shape[0:2])))
            plt.axis('off')
            plt.subplot(222)
            plt.imshow(j,cmap='gray')
            plt.title("{}x{}".format(*tuple(j.shape[0:2])))
            plt.axis('off')
            plt.subplot(223)
            plt.imshow(l,cmap='gray')
            plt.title("{}x{}".format(*tuple(l.shape[0:2])))
            plt.axis('off')
            plt.subplot(224)
            plt.imshow(m,cmap='gray')
            plt.title("{}x{}".format(*tuple(m.shape[0:2])))
            plt.axis('off')
            plt.show()
    # test_path = os.path.join(root_path,"test",task_list[2])
    # test_list = os.listdir(test_path)
    # start = time.time()
    # data,mask = read_test_data(test_path,test_list)
    # for one_patient_data,one_patient_mask in zip(data,mask):
    #     batch_object = test_batch(one_patient_data,one_patient_mask,7)
    #     while(1):
    #         one_batch,flag = batch_object.get_batch(1)
    #         if(not flag):
    #             break
    #         one_batch_data,one_batch_mask = one_batch[0],one_batch[1]
    #         print(one_batch_data,one_batch_mask)
    #         plt.subplot(121)
    #         plt.imshow(one_batch_data[2,:,:,0],cmap='gray')
    #         plt.axis('off')
    #         plt.subplot(122)
    #         plt.imshow(np.argmax(one_batch_mask[2],axis=-1),cmap='gray')
    #         plt.axis('off')

            