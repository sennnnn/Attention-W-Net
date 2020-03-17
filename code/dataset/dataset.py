import os
import sys
sys.path.append("..")
import random
import numpy as np

from util import readImage,readNiiAll,saveInfo
from arg_parser import arg_parser

nii_root_path = r'E:\dataset\zhongshan_hospital\CSTRO'
npzy_root_path = r'.'

target_list = ['Thoracic_OAR', 'HaN_OAR', 'Lung_GTV', 'Naso_GTV']

def process_train_dataset(target):

    nii_train_path = os.path.join(nii_root_path, 'train', target)
    npzy_train_path = os.path.join(npzy_root_path, 'train', target)

    _list = os.listdir(nii_train_path)
    _list = sorted(_list, key=lambda x: int(x))
    for one in _list:
        index = 0
        data_path = os.path.join(nii_train_path, one, 'data.nii')
        label_path = os.path.join(nii_train_path, one, 'label.nii')
        npzy_patient_path = os.path.join(npzy_train_path, one)
        data_array = readImage(data_path)
        label_array = readImage(label_path)
        
        if(not os.path.exists(npzy_train_path)):
            os.makedirs(npzy_train_path, 0x777)
        
        for slice_d,slice_l in zip(data_array, label_array):
            slice = np.stack([slice_d, slice_l], axis=0)
            np.save('{}/{}-{}.npy'.format(npzy_train_path, one, index), slice)
            index += 1
        print('doing: {}/{}'.format(one, len(_list)))

    print('tarin nii dataset done!!')

def process_test_dataset(target):

    nii_test_path = os.path.join(nii_root_path, 'test', target)
    npzy_test_path = os.path.join(npzy_root_path, 'test', target)

    _list = os.listdir(nii_test_path)
    _list = sorted(_list, key=lambda x: int(x))
    for one in _list:
        index = 0
        data_path = os.path.join(nii_test_path, one, 'data.nii')
        label_path = os.path.join(nii_test_path, one, 'label.nii')
        npzy_patient_path = os.path.join(npzy_test_path, one)
        info_dict = {}
        info_dict['spacing'],info_dict['origin'],data_array = readNiiAll(data_path)
        label_array = readImage(label_path)

        if(not os.path.exists(npzy_patient_path)):
            os.makedirs(npzy_patient_path, 0x777)
        
        for slice_d,slice_l in zip(data_array, label_array):
            slice = np.stack([slice_d, slice_l], axis=0)
            np.save('{}/{}.npy'.format(npzy_patient_path, index), slice)
            index += 1
        saveInfo(info_dict, '{}/nii_meta_info.txt'.format(npzy_patient_path))
        print('patient:{}/{}'.format(one, len(_list)))

    print('test nii dataset done!!')

def train_dataset_txt_generate(target):
    f = open('{}_train_dataset.txt'.format(target), 'w')
    npzy_train_path = os.path.join(npzy_root_path, 'train', target)
    npzy_list = os.listdir(npzy_train_path)
    f.write('# slice count:{}\n'.format(len(npzy_list)))
    for npzy in npzy_list:
        npzy_path = os.path.join('dataset', npzy_train_path, npzy)
        f.write(npzy_path + '\n')
    f.close()

def test_dataset_txt_generate(target):
    f = open('{}_test_dataset.txt'.format(target), 'w')
    npzy_test_path = os.path.join(npzy_root_path, 'test', target)
    patient_list = os.listdir(npzy_test_path)
    for patient in patient_list:
        patient_path = os.path.join(npzy_test_path, patient)
        npzy_list = os.listdir(patient_path)
        npzy_list = [x for x in npzy_list if(os.path.splitext(x)[1] != '.txt')]
        npzy_list = sorted(npzy_list, key=lambda x: int(os.path.splitext(x)[0]))
        npzy_list_length = len(npzy_list)
        patient_nii_meta_info_path = os.path.join('dataset', patient_path, 'nii_meta_info.txt')
        f.write('[patient:{}]\nslice_number:{}\n{}\n'.format(patient, npzy_list_length, \
                patient_nii_meta_info_path))
        for npzy in npzy_list:
            npzy_path = os.path.join('dataset', patient_path, npzy)
            f.write(npzy_path + '\n')
        f.write('\n')
    f.close()

if __name__ == "__main__":

    a = arg_parser()

    task_map = \
    {
        '--train': \
            {
                'txt' : 0,
                'npzy' : 1
            },

        '--test' : \
            {
                'txt' : 2,
                'npzy' : 3
            }
    }

    a.add_map('--train', task_map['--train'])
    a.add_map('--test', task_map['--test'])

    target_dict = \
    {
        '--target': \
            {
                '0': 'Thoracic_OAR',
                '1': 'HaN_OAR',
                '2': 'Lung_GTV',
                '3': 'Naso_GTV'
            }
    }

    a.add_map('--target', target_dict['--target'])

    ret_dict = a()

    eval_list = []

    target = ret_dict['--target']

    if('--test' in list(ret_dict.keys())):
        if(ret_dict['--test'] == 2):
            eval_list.append(test_dataset_txt_generate)
        
        if(ret_dict['--test'] == 3):
            eval_list.append(process_test_dataset)

    if('--train' in list(ret_dict.keys())):
        if(ret_dict['--train'] == 0):
            eval_list.append(train_dataset_txt_generate)
        
        if(ret_dict['--train'] == 1):
            eval_list.append(process_train_dataset)

    [x(target) for x in eval_list]
