import os
import tensorflow as tf

from test import test_all
from train import train_all
from arg_parser import args_process
from model.model_util import load_graph
from process import train_valid_generator,test_generator
from util import read_train_valid_data,read_test_data,get_newest,dict_load

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

ret_dict = args_process()

input_shape_dict = {'HaN_OAR': (224, 224), 'Thoracic_OAR': (256, 256), 'Naso_GTV': (224, 224), 'Lung_GTV': (256, 256)}
crop_range_dict = {'HaN_OAR': {'x': (160, 384), 'y': (160, 384)},
                   'Thoracic_OAR': {'x': (117, 417), 'y': (99, 399)},
                   'Lung_GTV': {'x': (117, 417), 'y': (99, 399)},
                   'Naso_GTV': {'x': (160, 384), 'y': (160, 384)}}
num_class_dict = {'HaN_OAR': 23, 'Thoracic_OAR': 7, 'Lung_GTV': 2, 'Naso_GTV': 2}

if(ret_dict['task'] == 'train'):

    # target selection
    target = ret_dict['target']

    # rarely changing options
    input_shape = input_shape_dict[target]
    crop_x_range = crop_range_dict[target]['x']
    crop_y_range = crop_range_dict[target]['y']
    resize_shape = (512, 512)
    num_class = num_class_dict[target]
    initial_channel = 64
    max_epoches = 200

    # usually changing options
    last = ret_dict['last']
    start_epoch = ret_dict['start_epoch']
    pattern = ret_dict['model_pattern']
    model_key = ret_dict['model']
    batch_size = 4
    learning_rate = 0.0001
    keep_prob = 0.1

    frozen_model_path = "build/{}-{}/frozen_model".format(model_key, target)
    ckpt_path = "build/{}-{}/ckpt".format(model_key, target)

    train_path_list,valid_path_list = read_train_valid_data("dataset/{}_train_dataset.txt".format(target), valid_rate=0.3, ifrandom=True)

    # input_shape is the shape of numpy array, but it isn't same as the opencv.
    # I hate opencv.
    train_batch_generator = train_valid_generator(train_path_list, len(train_path_list), True, batch_size, num_class,\
                                            input_shape, resize_shape, crop_x_range, crop_y_range, True, True, 15)

    valid_batch_generator = train_valid_generator(valid_path_list, len(valid_path_list), True, batch_size, num_class, \
                                            input_shape, resize_shape, crop_x_range, crop_y_range, False, False, 0)

    # load graph or init graph
    train_object = train_all(last, pattern, model_key, frozen_model_path, ckpt_path, \
                             num_class, initial_channel, target)

    train_object.training(learning_rate, max_epoches, 5, \
                          start_epoch, train_batch_generator, valid_batch_generator, 3, 5, keep_prob, config)

elif(ret_dict['task'] == 'test'):

    # target selection
    target = ret_dict['target']

    # rarely changing options
    input_shape = input_shape_dict[target]
    crop_x_range = crop_range_dict[target]['x']
    crop_y_range = crop_range_dict[target]['y']
    resize_shape = (512, 512)
    num_class = num_class_dict[target]

    # usually changing options
    keep_prob = 0.1
    batch_size = 2
    model_key = ret_dict['model']

    frozen_model_path = "build/{}-{}/frozen_model".format(model_key, target)

    frozen_model_name = os.path.basename(get_newest(frozen_model_path))

    graph = load_graph(get_newest(frozen_model_path))

    test_path_list = read_test_data("dataset/{}_test_dataset.txt".format(target))

    test_batch_generator = test_generator(test_path_list, batch_size, num_class, input_shape, resize_shape, \
                                          crop_x_range, crop_y_range)

    test_object = test_all(graph, model_key, frozen_model_path, frozen_model_name, num_class, target)
    test_object.testing(keep_prob, test_batch_generator, config, True)
else:
    print('Sorry,{} isn’t a valid option'.format(sys.argv[1]))
    exit()

