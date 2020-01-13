import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

from util import one_hot,get_newest,readImage,np_dice_index,grayToRgb,stringToHex,one_hot

test_real_path = r"E:\dataset\zhongshan_hospital\cstro\test\Thoracic_OAR"
test_process_path = "test_result/77_0.925.pb_process"
test_raw_path = "test_result/77_0.925.pb_raw"
test_list = os.listdir(test_real_path)
one_test = "1"
index = 52
# font = {}
font={
#      'family':'serif',
#      'style':'italic',
#      'weight':'normal',
    #    'color':'white',
#       'size':16
}
one_test_raw_path = os.path.join(test_raw_path,one_test,"test_label.nii.gz")
one_test_process_path = os.path.join(test_process_path,one_test,"test_label.nii.gz")
one_real_test_path = os.path.join(test_real_path,one_test,"label.nii.gz")
raw = readImage(one_test_raw_path);real = readImage(one_real_test_path);process = readImage(one_test_process_path)
one = plt.figure(1)
plt.subplot(131)
plt.imshow(grayToRgb(real[index]),cmap='gray')
plt.title('GroundTruth',font)
plt.axis('off')
plt.subplot(132)
plt.imshow(grayToRgb(raw[index]),cmap='gray')
plt.title('No-process dice:%.3f'%(np_dice_index(one_hot(raw[index],7),one_hot(real[index],7))),font)
plt.axis('off')
plt.subplot(133)
plt.imshow(grayToRgb(process[index]),cmap='gray')
plt.title('After-process dice:%.3f'%(np_dice_index(one_hot(raw[index],7),one_hot(process[index],7))),font)
plt.axis('off')
plt.show()
