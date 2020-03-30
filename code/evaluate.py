import os

from util import get_newest,readImage,one_hot,average
from model.loss_metric import *
from arg_parser import arg_parser

a = arg_parser()

metric_dict = {'Precision': Precision, 'IoU': IOU, 'DSC': DSC};metric_keys = list(metric_dict.keys())
model_dict = {'unet':'Unet', 'unet_se':'Unet-SE', 'uplus':'Unet++', \
              'r2u':'R2Unet', 'ce':'CEnet', 'wnet':'Wnet', 'wnet_r':'Wnet_raw', \
              'att':'Attention-Unet', 'att_se':'Attention-Unet-SE', 'attwnet':'Attention-Wnet', 'att_old': 'Attention-Unet-old'}
target_dict = {'HN_OAR': 'HaN_OAR', 'Lu_OAR': 'Thoracic_OAR', 'HN_GTV': 'Naso_GTV', 'Lu_GTV': 'Lung_GTV'}
num_class_dict = {'HaN_OAR': 23, 'Thoracic_OAR': 7, 'Lung_GTV': 2, 'Naso_GTV': 2}

a.add_map('--model', model_dict)
a.add_map('--target', target_dict)

ret_dict = a()

model_key = ret_dict['--model']
target = ret_dict['--target']
num_class = num_class_dict[target]

label_root_path = os.path.join(r'E:\dataset\zhongshan_hospital\CSTRO\test', 'HaN_OAR')
test_list = os.listdir(label_root_path)
predict_root_path = get_newest(os.path.join('build', '{}-{}'.format(model_key, target), 'test_result'))

metric_list = {key:[] for key in metric_keys}
metric_channelwise_list = {key:[] for key in metric_keys}
f_dict = {key:open('{}/{}.txt'.format(predict_root_path, key), 'w') for key in metric_keys}

for key in metric_keys:
    f = f_dict[key]
    f.write('name ')
    for i in range(1, num_class):
        f.write('OAR%d '%(i))
    f.write('average\n')

count = 0
for patient in test_list:
    predict_path = os.path.join(predict_root_path, patient, 'predict.nii.gz')
    label_path = os.path.join(label_root_path, patient, 'label.nii.gz')
    
    predict = readImage(predict_path).astype(np.uint8)
    label = readImage(label_path)

    predict = one_hot(predict)
    label = one_hot(label)

    for key in metric_keys:
        f = f_dict[key]
        temp_list = []
        temp_string = 'patient{} '.format(patient)
        for i in range(1, num_class):
            metric = metric_dict[key](label[..., i], predict[..., i])
            temp_string += '%.4f '%(metric)
            temp_list.append(metric)
        avg = average(temp_list)
        temp_string += '%.4f\n'%(avg)
        metric_channelwise_list[key].append(temp_list)
        metric_list[key].append(avg)
        f.write(temp_string)
    
    count += 1
    print('patient{} done {}/{}'.format(patient, count, len(test_list)))

for key in metric_keys:
    f = f_dict[key]
    temp_string = 'average '
    for i in range(1, num_class):
        temp_string += '%.4f '%(average([sub[i-1] for sub in metric_channelwise_list[key]]))
    temp_string += '%.4f\n'%(average(metric_list[key]))
    f.write(temp_string)
    f.close()