import os
import cv2
import time
import random
import numpy as np
import SimpleITK as stk
import matplotlib.pyplot as plt
from util import one_hot

task_list = ["HaN_OAR","Naso_GTV","Thoracic_OAR","Lung_GTV"]
root_path = r"E:\dataset\datasets\cstro"

def readInformation(nii_path):
    image = stk.ReadImage(nii_path)
    return image.GetSpacing(),image.GetOrigin(),image.GetPixel()

def saveAsNiiGz(numpy_array,nii_path,origin,spacing,pixel):
    image = stk.GetImageFromArray(numpy_array)
    image.SetSpacing(spacing);image.SetOrigin(origin);image.SetPixel(pixel)
    stk.WriteImage(image,nii_path)
    print("nii file is saved as {}".format(nii_path))

def get_batch(data_block,mask_block,batch_size,flip,angle):
    length = data_block.shape[0]
    index  = random.randint(0,length-batch_size)
    batch_data = []
    batch_mask = []
    for i in range(index,index+batch_size):
        temp_slice_data = _preprocess_oneslice(data_block[i],flip,angle,"data")
        temp_slice_mask = _preprocess_oneslice(mask_block[i].flip,angle,"mask")
        batch_data.append(temp_slice_data)
        batch_mask.append(temp_slice_mask)
    return np.expand_dims(np.array(batch_data),-1),one_hot(np.array(batch_mask),7)

def readImage(nii_path):
    image = stk.ReadImage(nii_path)
    return stk.GetArrayFromImage(image)

def _preprocess_oneslice(one_slice,filp,angle,flag="data"):
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
    # 裁剪，裁剪是最优先的，是数据简化的最有效方式
    out = out[np.ix_(range(140,396),range(122,378))]
    # 获得旋转矩阵
    if(angle!=0):
        h,w = out.shape[:2]
        center = (w//2,h//2)
        random_angle = random.randint(-1*angle,angle)
        rotate_matrix = cv2.getRotationMatrix2D(center,random_angle,1)
        # 旋转
        out = cv2.warpAffine(out, rotate_matrix, (w,h))
    
    if(filp==True):
        random_state = random.randint(-6,2)
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

def read_train_data(train_path,train_list):
    """
    载入所有的训练数据
    """
    data_list = []
    mask_list = []
    for patient in train_list:
        data = readImage(os.path.join(train_path,patient,'data.nii'))
        mask = readImage(os.path.join(train_path,patient,'label.nii'))
        for i,j in zip(data,mask):
            # 裁剪，裁剪是最优先的，是数据简化的最有效方式
            one_data_slice = i[np.ix_(range(140,396),range(122,378))]
            one_mask_slice = j[np.ix_(range(140,396),range(122,378))]
            data_list.append(one_data_slice)
            mask_list.append(one_mask_slice)
    return np.array(data_list),np.array(mask_list)

def read_test_data(test_path,test_list):
    """
    载入所有的测试数据，测试数据不需要做数据预处理
    """
    data_patients_list = []
    mask_patients_list = []
    for patient in test_list:
        data = readImage(os.path.join(test_path,patient,'data.nii'))
        mask = readImage(os.path.join(test_path,patient,'label.nii'))
        one_patient_data = []
        one_patient_mask = []
        for i,j in zip(data,mask):
            one_data_slice = _preprocess_oneslice(i,False,0,"data")
            one_mask_slice = _preprocess_oneslice(j,False,0,"label")
            one_patient_data.append(one_data_slice)
            one_patient_mask.append(one_mask_slice)
        data_patients_list.append(np.array(one_patient_data))
        mask_patients_list.append(np.array(one_patient_mask))
    return np.array(data_patients_list),np.array(mask_patients_list)

class test_batch(object):
    def __init__(self,one_patient_data,one_patient_mask):
        self.data = one_patient_data
        self.mask = one_patient_mask
        self.length = one_patient_data.shape[0]
        self.index = 0

    def __do(self,data,mask,start,end):
        return np.expand_dims(data,axis=-1)[start:end],one_hot(mask,7)[start:end]

    def get_batch(self,batch_size):
        self.index += batch_size
        # 当已经把一个病人遍历完了的时候那么就会输出空
        if(self.index >= self.length):
            return [],False

        if(self.index + batch_size > self.length):
            start = self.index
            end = self.length
            print(start,end)
            return self.__do(self.data,self.mask,start,end),True
        else:
            start = self.index
            end = self.index + batch_size
            return self.__do(self.data,self.mask,start,end),True

if __name__ == "__main__":
    train_path = os.path.join(root_path,"train",task_list[2])
    train_list = os.listdir(train_path)
    start = time.time()
    data,mask = read_train_data(train_path,train_list,1,15)
    end = time.time()
    print("spend time:%.2fs\ndata_shape:{} mask_shape:{}".format(data.shape,mask.shape)%(end-start))
    # data = readImage(os.path.join(train_path,train_list[0],"data.nii"))
    # mask = readImage(os.path.join(train_path,train_list[0],"label.nii"))
    while(1):
        batch_x,batch_y = get_batch(data,mask,8)
        print(batch_x.shape," ",batch_y.shape)
        for i,j in zip(batch_x,batch_y):
            one_data_slice = i[...,0]
            one_mask_slice = np.argmax(j,axis=-1)
            plt.subplot(121)
            plt.imshow(one_data_slice,cmap='gray')
            plt.axis('off')
            plt.subplot(122)
            plt.imshow(one_mask_slice,cmap='gray')
            plt.axis('off')
            plt.show()
