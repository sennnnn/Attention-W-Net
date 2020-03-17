import os
import cv2
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from util import one_hot,readImage,loadInfo

def _preprocess_oneslice(one_slice, tran, filp, angle):
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
    # get the rotation matrix.
    if(angle!=0):
        center = (w//2, h//2)
        random_angle = random.randint(-1*angle, angle)
        rotate_matrix = cv2.getRotationMatrix2D(center, random_angle, 1)
        # rotation
        out_data = cv2.warpAffine(out_data, rotate_matrix, (w, h))
        out_mask = cv2.warpAffine(out_mask, rotate_matrix, (w, h))
    if(tran==True):
        x_shift = random.randint(-10, 10)
        y_shift = random.randint(-10, 10)
        # get the pan matrix.
        trans_matrix = np.array([[1, 0, x_shift],[0, 1, y_shift]],dtype=np.float32)
        out_data = cv2.warpAffine(out_data, trans_matrix, (w, h))
        out_mask = cv2.warpAffine(out_mask, trans_matrix, (w, h))

    if(filp==True):
        random_state = random.randint(-6,2)
        if(random_state>=0):
            out_data = cv2.flip(out_data, random_state-1)
            out_mask = cv2.flip(out_mask, random_state-1)
        # horizontal flip.
        # out = cv2.flip(out,1)
        # vertical flip.
        # out = cv2.flip(out,0)
        # vertival & horizontal flip.
        # out = cv2.flip(out,-1)
    
    # max-min normalization or z-score Standardization, Only for data.
    """
    z-score Standardization
    """
    mean,stdDev = cv2.meanStdDev(out_data)
    mean,stdDev = mean[0][0],stdDev[0][0]
    out_data = (out_data-mean)/stdDev

    return (out_data,out_mask)

def length_norm_crop(slice, general_shape, crop_x_range, crop_y_range):
    slice = cv2.resize(slice, general_shape)
    slice = slice[np.ix_(range(*crop_x_range), range(*crop_y_range))]

    return slice

class train_valid_generator(object):
    def __init__(self, path_list, slice_count, ifrandom, batch_size, num_class,\
                 input_shape, resize_shape, crop_x_range, crop_y_range, tran, flip, angle):
        self.path_list = path_list
        self.ifrandom = ifrandom
        self.slice_count = slice_count
        self.epoch_steps = slice_count//batch_size
        self.batch_size = batch_size
        self.num_class = num_class
        self.input_shape = input_shape
        self.resize_shape = resize_shape
        self.crop_x_range = crop_x_range
        self.crop_y_range = crop_y_range
        self.tran = tran
        self.flip = flip
        self.angle = angle

    def __iter__(self):
        temp = self.path_list
        if(self.ifrandom):
            random.shuffle(temp)
        
        count = 0
        
        batch = ([], [])

        for one_record in temp:
            count += 1
            data_label_block = np.load(one_record)
            
            data_label_block = self.process(data_label_block)

            batch[0].append(data_label_block[0]);batch[1].append(data_label_block[1])
            
            if(count == self.batch_size):
                count = 0
                yield batch
                batch[0].clear()
                batch[1].clear()
    
    def process(self, sequence_block):
        i,j = sequence_block[0],sequence_block[1]
        i = i.astype(np.float32)
        j = j.astype(np.uint8)
        i = length_norm_crop(i, self.resize_shape, self.crop_x_range, self.crop_y_range)
        j = length_norm_crop(j, self.resize_shape, self.crop_x_range, self.crop_y_range)
        i = cv2.resize(i, (self.input_shape))
        j = cv2.resize(j, (self.input_shape))
        i,j = _preprocess_oneslice((i, j), self.tran, self.flip, self.angle)
        i = np.expand_dims(i, axis=-1)
        j = one_hot(j, self.num_class)

        return i,j

    def epochwise_iter(self):
        while(True):
            yield from self.__iter__()

class test_generator(object):
    def __init__(self, path_list, batch_size, num_class, input_shape, resize_shape,\
                 crop_x_range, crop_y_range):
        self.path_list = path_list
        self.batch_size = batch_size
        self.num_class = num_class
        self.input_shape = input_shape
        self.resize_shape = resize_shape
        self.crop_x_range = crop_x_range
        self.crop_y_range = crop_y_range

    def _path_resolve(self):
        self.test_data = {}
        patient_name = None
        count_next_flag = False
        info_next_flag = False
        for line in self.path_list:
            if(line == '\n'):
                continue
            line = line.strip()
            if(count_next_flag):
                self.test_data[patient_name]['slice_count'] = int(line.split(':')[1])
                count_next_flag = False
                info_next_flag = True
                continue
            if(info_next_flag):
                self.test_data[patient_name]['nii_info'] = loadInfo(line)
                info_next_flag = False
                continue
            if(line[0] == '['):
                patient_name = line.split(':')[1]
                patient_name = patient_name.replace(']', '')
                self.test_data[patient_name] = {}
                self.test_data[patient_name]['npzy_paths'] = []
                count_next_flag = True
                continue

            self.test_data[patient_name]['npzy_paths'].append(line)
    
    def process(self, sequence_block):
        i,j = sequence_block[0],sequence_block[1]
        i = i.astype(np.float32)
        j = j.astype(np.uint8)
        i = length_norm_crop(i, self.resize_shape, self.crop_x_range, self.crop_y_range)
        i,j = _preprocess_oneslice((i, j), False, False, 0)
        i = cv2.resize(i, (self.input_shape))
        i = np.expand_dims(i, axis=-1)
        j = one_hot(j, self.num_class)

        return i,j

    def __iter__(self, patient_name):
        npz_list = self.test_data[patient_name]['npzy_paths']
        flag = 0
        batch = ([], [])

        for one_record in npz_list:
            flag += 1
            data_label_block = np.load(one_record)
            data_label_block = self.process(data_label_block)

            batch[0].append(data_label_block[0]);batch[1].append(data_label_block[1])
            
            if(flag == self.batch_size):
                flag = 0
                yield batch
                batch[0].clear();batch[1].clear()

        if(flag != 0):
            yield batch

    def patientwise_iter(self):
        self._path_resolve()
        iter_list = []
        for patient_name in self.test_data.keys():
            yield (patient_name, self.test_data[patient_name]['nii_info'], self.__iter__(patient_name))

def recover(patient_mask_predict, shape, ifprocess, num_class, crop_x_range, crop_y_range, resize_shape):
    length = len(patient_mask_predict)
    w = crop_x_range[1]-crop_x_range[0]
    h = crop_y_range[1]-crop_y_range[0]
    out = np.array([cv2.resize(x.astype(np.uint8), (h, w)) for x in patient_mask_predict], dtype=np.uint8)
    temp = np.zeros(shape=shape, dtype=np.uint8)
    for i in range(length):
        temp_slice = np.zeros(shape=resize_shape, dtype=np.uint8)
        temp_slice[np.ix_(range(*crop_x_range), range(*crop_y_range))] = out[i]
        temp[i] = cv2.resize(temp_slice, tuple(temp[i].shape))
    # After postprocessing...
    temp = one_hot(temp, num_class)
    if(ifprocess):
        temp = np.array([after_process(temp[i]) for i in range(temp.shape[0])])

    return temp

def dilate(src, kernel_size=(3, 3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    dst = cv2.dilate(src, kernel, iterations=2)

    return dst

def erode(src, kernel_size=(3, 3)):
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    dst = cv2.erode(src, kernel, iterations=1)

    return dst

def left_lung_after_process(src, benchmark):
    # 输入的是一张slice，src为左肺切片，benchmark为右肺切片
    # 即将右肺预测到左肺的部分给去掉
    begin = 0
    end = 0
    # 首先获得搜索开始的y轴下标
    for i in range(src.shape[1]):
        if(np.max(src[:,i]) != 0):
            begin = i
            break

    for j in range(begin, src.shape[1]):
        if(np.max(src[:,j]) == 0):
            end = j
            break

    for k in range(begin, end):
        for x_index in range(src.shape[0]):
            if(benchmark[x_index,k] != 0):
                src[x_index,k] = 2

    return src
    
def right_lung_after_process(src, benchmark):
    # 输入的是一张slice，src为右肺切片，benchmark为左肺切片
    # 即将左肺预测到右肺的部分给去掉
    begin = 0
    end = 0
    # 首先获得搜索开始的y轴下标 
    for i in range(src.shape[1])[::-1]:
        if(np.max(src[:,i]) != 0):
            begin = i
            break

    for j in range(begin)[::-1]:
        if(np.max(src[:,j]) == 0):
            end = j
            break

    for k in range(end, begin)[::-1]:
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
    one_slice_mid_dst = np.expand_dims(one_slice[...,0], axis=-1)
    # 腐蚀处理
    for j in range(1,one_slice.shape[-1]):
        temp = np.expand_dims(erode(one_slice[...,j]), axis=-1)
        one_slice_mid_dst = np.concatenate([one_slice_mid_dst, temp], axis=-1)
    # 左右肺排查，已经膨胀处理，这是针对肺部的操作
    # 测试出来的有效面积
    # if(task == "Thoracic_OAR"):
    #     if(findMaxContours(one_slice_mid_dst[...,2]) > 800):
    #         one_slice_mid_dst[...,2] = left_lung_after_process(one_slice_mid_dst[...,2],one_slice_mid_dst[...,1])
    #     if(findMaxContours(one_slice_mid_dst[...,1]) > 800):
    #         one_slice_mid_dst[...,1] = right_lung_after_process(one_slice_mid_dst[...,1],one_slice_mid_dst[...,2])
    
    one_slice_dst = np.expand_dims(one_slice_mid_dst[...,0], axis=-1)
    for j in range(1,one_slice.shape[-1]):
        temp = np.expand_dims(dilate(one_slice_mid_dst[...,j]), axis=-1)
        one_slice_dst = np.concatenate([one_slice_dst, temp], axis=-1)
    
    return one_slice_dst

if __name__ == "__main__":
    """
    猛然意识到之前写得代码实在是太差了。
    顺带一提，正在听：Если завтра война 
    """

            