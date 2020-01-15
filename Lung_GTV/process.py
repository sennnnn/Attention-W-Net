import os
import cv2
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from util import one_hot,readImage

task_list = ["HaN_OAR","Naso_GTV","Thoracic_OAR","Lung_GTV"]
root_path = r"E:\dataset\zhongshan_hospital\cstro"

def _preprocess_oneslice(one_slice,filp,angle,random_state=None,flag="data"):
    """
    cstro比赛统一为512*512的规格
    主要项目：裁剪、翻转、旋转
    args:
    one_slice : A numpy mat and it shapes like 512x512 and the data type is np.float32.
    
    return:
    数据增强完毕的单张切片
    """
    out = one_slice
    if(flag=="data"):
        out = out.astype(np.float32)
        out = out + abs(np.min(out))
    # 获得旋转矩阵
    if(angle!=0):
        h,w = out.shape[:2]
        center = (w//2,h//2)
        random_angle = random.randint(-1*angle,angle)
        rotate_matrix = cv2.getRotationMatrix2D(center,random_angle,1)
        # 旋转
        out = cv2.warpAffine(out, rotate_matrix, (w,h))
    
    if(filp==True):
        if(random_state>=0):
            out = cv2.flip(out, random_state-1)
        # 水平翻转
        # out = cv2.flip(out,1)
        # 垂直翻转
        # out = cv2.flip(out,0)
        # 水平垂直翻转
        # out = cv2.flip(out,-1)
    
    # 最大最小归一化、函数转化归一化、z-score标准化，仅针对数据而不是标签
    if(flag=="data"):
        """
        z-score 标准化
        """
        mean,stdDev = cv2.meanStdDev(out)
        mean,stdDev = mean[0][0],stdDev[0][0]
        out = (out-mean)/stdDev
    return out

def read_train_data(train_path,train_list,input_shape):
    """
    载入所有的训练数据
    """
    data_list = []
    mask_list = []
    for one_patient in train_list:
        data = readImage(os.path.join(train_path,one_patient,'data.nii'))
        mask = readImage(os.path.join(train_path,one_patient,'label.nii'))
        for i,j in zip(data,mask):
            i = i[np.ix_(range(117,417),range(99,399))]
            j = j[np.ix_(range(117,417),range(99,399))]
            i = cv2.resize(i.astype(np.float32),input_shape)
            j = cv2.resize(j,input_shape)
            data_list.append(i)
            mask_list.append(j)
    return np.array(data_list),np.array(mask_list)

def epoch_read(data,mask):
    # 每个epoch重新读取一次数据集，将正负样本在每个epoch下保持一定比例
    # 为了让正负样本均衡，而又不让负样本的多样性太单一，目前训练集的比例为positive:negetive ≈ 400:3900
    # 首先提取出有效训练数据，epoch28之前每个epoch正样本是随机取所有正样本的3/4，而负样本是取之前所有负样本1/5
    # epoch28之后调整正负样本比例，负样本应该更少，因为测试结果发现预测的部分太少了。
    # epoch28之后正样本为随机在所有样本里面取3/4，负样本随机在所有样本里面取3/20
    data_list = []
    mask_list = []
    # 正样本中提取一部分，负样本中提取一部分
    for i,j in zip(data,mask):
        if(np.max(j) == 0):
            flag = random.randint(1,20)
            if(flag < 4):
                data_list.append(i)
                mask_list.append(j)
        else:
            flag = random.randint(1,4)
            if(flag < 3):
                data_list.append(i)
                mask_list.append(j)
    return np.array(data_list),np.array(mask_list)

def recover_softmax(patient_softmax_mask_predict,shape,num_class):
    length = len(patient_softmax_mask_predict)
    out = np.array([cv2.resize(x.astype(np.float32), (300,300)) for x in patient_softmax_mask_predict],dtype=np.float32)
    temp = np.zeros(shape=(*shape,num_class), dtype=np.float32)
    temp[np.ix_(range(0,length),range(117,417),range(99,399),range(0,num_class))] = out
    return temp

def recover(patient_mask_predict,shape,ifprocess,num_class):
    # 还原
    length = len(patient_mask_predict)
    out = np.array([cv2.resize(x.astype(np.uint8), (300,300)) for x in patient_mask_predict],dtype=np.uint8)
    temp = np.zeros(shape=shape, dtype=np.uint8)
    temp[np.ix_(range(0,length),range(117,417),range(99,399))] = out
    # 后处理之后
    temp = one_hot(temp, num_class)
    if(ifprocess):
        temp = np.array([after_process(x) for x in temp])
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

def after_process(one_slice):
    # after one_hot multi slice block
    # 这里只是用到了简单的形态学腐蚀，那么我们再假设，左肺最右点不可能有右肺，右肺最左点不可能有左肺
    # 搜索左肺最右点，即y最大点
    # 输入进来的为病人的完整block，且已经one_hot编码
    one_slice = one_slice
    one_slice_dst = np.expand_dims(one_slice[...,0],axis=-1)
    # 腐蚀处理
    for j in range(1,one_slice.shape[-1]):
        temp = np.expand_dims(erode(one_slice[...,j]),axis=-1)
        one_slice_dst = np.concatenate([one_slice_dst,temp],axis=-1)
    # 左右肺排查，已经膨胀处理
    # 测试出来的有效面积
    if(findMaxContours(one_slice_dst[...,2]) > 800):
        one_slice_dst[...,2] = left_lung_after_process(one_slice_dst[...,2],one_slice_dst[...,1])
    if(findMaxContours(one_slice_dst[...,1]) > 800):
        one_slice_dst[...,1] = right_lung_after_process(one_slice_dst[...,1],one_slice_dst[...,2])
    one_slice_dst[...,2] = dilate(one_slice_dst[...,2])
    one_slice_dst[...,1] = dilate(one_slice_dst[...,1])
    return one_slice_dst

class train_batch(object):
    def __init__(self,data_block,mask_block,flip,angle,num_class):
        self.data = data_block
        self.mask = mask_block
        self.flip = flip
        self.angle = angle
        self.num_class = num_class
        self.length = data_block.shape[0]

    def get_batch(self,batch_size):
        length = self.length
        index  = random.randint(0,length-batch_size)
        batch_data = []
        batch_mask = []
        random_state=None
        for i in range(index,index+batch_size):
            if(self.flip == True):
                random_state = random.randint(-9,2)
            # 裁剪，裁剪是最优先的，是数据简化的最有效方式
            one_data_slice = self.data[i]
            one_mask_slice = self.mask[i]
            temp_slice_data = _preprocess_oneslice(one_data_slice,self.flip,self.angle,random_state,"data")
            temp_slice_mask = _preprocess_oneslice(one_mask_slice,self.flip,self.angle,random_state,"mask")
            batch_data.append(temp_slice_data)
            batch_mask.append(temp_slice_mask)
        return np.expand_dims(np.array(batch_data),-1),one_hot(np.array(batch_mask),self.num_class)

class test_batch(object):
    def __init__(self,one_patient_data,one_patient_mask,num_class,input_shape):
        self.data = one_patient_data
        self.mask = one_patient_mask
        self.num_class = num_class
        self.length = one_patient_data.shape[0]
        self.index = 0
        self.input_shape = input_shape

    def __do(self,data,mask,start,end):
        batch_data = []
        batch_mask = []
        data = data[start:end]
        mask = mask[start:end]
        for i,j in zip(data,mask):
            i = i[np.ix_(range(117,417),range(99,399))]
            j = j[np.ix_(range(117,417),range(99,399))]
            i = cv2.resize(i.astype(np.float32),self.input_shape)
            j = cv2.resize(j,self.input_shape)
            i = _preprocess_oneslice(i,False,0,flag="data")
            j = _preprocess_oneslice(j,False,0,flag="mask")
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

if __name__ == "__main__":
    # train batch get example
    """
    6
    """
            