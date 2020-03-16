import os
import time
import random
import datetime
import numpy as np
import SimpleITK as stk

def readNiiAll(nii_path):
    image = stk.ReadImage(nii_path)
    array = stk.GetArrayFromImage(image)

    return image.GetSpacing(),image.GetOrigin(),array

def readImage(nii_path):
    image = stk.ReadImage(nii_path)

    return stk.GetArrayFromImage(image)

def saveAsNiiGz(numpy_array, nii_path, spacing, origin):
    image = stk.GetImageFromArray(numpy_array)
    image.SetSpacing(spacing);image.SetOrigin(origin)
    stk.WriteImage(image,nii_path)
    print('nii file is saved as {}'.format(nii_path))

def saveInfo(info, txt_path):
    f = open(txt_path, 'w')
    f.write(str(info))
    f.close()

def loadInfo(txt_path):
    f = open(txt_path, 'r')
    
    return eval(f.read())

def get_newest(dir_path):
    file_list = os.listdir(dir_path)
    newest_file = os.path.join(dir_path,file_list[0])
    for filename in file_list:
        one_file = os.path.join(dir_path, filename)
        if(get_ctime(newest_file) < get_ctime(one_file)):
            newest_file = one_file

    return newest_file 

def get_ctime(file_path, ifstamp=True):
    if(ifstamp):
        return os.path.getctime(file_path)
    else:
        timeStruct = time.localtime(os.path.getctime(file_path))
        return time.strftime("%Y-%m-%d %H:%M:%S",timeStruct)

def ifsmaller(price_list, price):
    if(len(price_list) == 0):
        return 0
    else:
        if(price <= price_list[-1]):
            return 1
        else:
            return 0

def iflarger(price_list, price):
    if(len(price_list) == 0):
        return 0
    else:
        if(price >= price_list[-1]):
            return 1
        else:
            return 0

def read_train_valid_data(dataset_path, valid_rate=0.2, ifrandom=True):
    f = open(dataset_path, 'r')
    info = f.readlines()
    path_list = info[1:]
    path_list = [line.strip() for line in path_list]
    
    # 计算训练集和验证集的条目数量
    all_count = int(info[0].split(':')[1])
    valid_count = int(all_count * valid_rate)

    # 为训练集和验证集分配数据条目
    if(ifrandom):
        random.shuffle(path_list)
    
    valid_path_list = path_list[:valid_count]
    train_path_list = path_list[valid_count:]

    return train_path_list,valid_path_list

def read_test_data(dataset_path):
    f = open(dataset_path, 'r')
    info = f.readlines()

    return info

def dict_save(dict_to_save, save_path):
    f = open(save_path, 'w')
    f.write(str(dict_to_save))
    f.close()

def dict_load(load_path):
    f = open(load_path, 'r')
    temp = eval(f.read())
    f.close()
    
    return temp

def average(iterable_object):
    length = 0
    summary = 0
    for i in iterable_object:
        length += 1
        summary += i
    
    return summary/length

def one_hot(nparray, depth=0, on_value=1, off_value=0):
    if depth == 0:
        depth = np.max(nparray) + 1
    # 深度应该符合one_hot条件，其实keras有to_categorical(data,n_classes,dtype=float..)弄成one_hot
    assert np.max(nparray) < depth, "the max index of nparray: {} is larger than depth: {}".format(np.max(nparray), depth)
    shape = nparray.shape
    out = np.ones((*shape, depth), np.uint8) * off_value
    indices = []
    for i in range(nparray.ndim):
        tiles = [1] * nparray.ndim
        s = [1] * nparray.ndim
        s[i] = -1
        r = np.arange(shape[i]).reshape(s)
        if i > 0:
            tiles[i-1] = shape[i-1]
            r = np.tile(r, tiles)
        indices.append(r)
    indices.append(nparray)
    out[tuple(indices)] = on_value

    return out