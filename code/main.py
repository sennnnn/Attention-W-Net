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

if(ret_dict['task'] == 'train'):
    # rarely changing options
    input_shape = (224, 384)
    crop_x_range = (152, 600)
    crop_y_range = (0, 768)
    resize_shape = (768, 768)
    num_class = 6
    initial_channel = 32
    max_epoches = 200

    # usually changing options
    last = ret_dict['last']
    start_epoch = ret_dict['start_epoch']
    pattern = ret_dict['model_pattern']
    model_key = ret_dict['model']
    batch_size = 4
    learning_rate = 0.00005
    keep_prob = 0.1
    sequence = ret_dict['sequence']

    frozen_model_path = "build/{}-{}/frozen_model".format(model_key, sequence)
    ckpt_path = "build/{}-{}/ckpt".format(model_key, sequence)

    train_path_list,valid_path_list = read_train_valid_data("dataset/train_dataset.txt", valid_rate=0.3, ifrandom=True)

    # input_shape is the shape of numpy array, but it isn't same as the opencv.
    # In fact, the numpy array will be input_shape[::-1], (384, 224)
    # I hate opencv.
    train_batch_generator = train_valid_generator(train_path_list, len(train_path_list), True, batch_size, num_class,\
                                            input_shape, resize_shape, crop_x_range, crop_y_range, True, True, 15)
    valid_batch_generator = train_valid_generator(valid_path_list, len(valid_path_list), True, batch_size, num_class, \
                                            input_shape, resize_shape, crop_x_range, crop_y_range, False, False, 0)

    # load graph or init graph
    train_object = train_all(last, pattern, model_key, frozen_model_path, ckpt_path, \
                             num_class, initial_channel, sequence)

    train_object.training(learning_rate, max_epoches, len(train_path_list)//batch_size, \
                        start_epoch, train_batch_generator, valid_batch_generator, 3, 5, keep_prob, config)
elif(ret_dict['task'] == 'test'):
    # rarely changing options
    input_shape = (224, 384)
    crop_x_range = (152, 600)
    crop_y_range = (0, 768)
    resize_shape = (768, 768)
    num_class = 6

    # usually changing options
    keep_prob = 0.1
    batch_size = 2
    sequence = ret_dict['sequence']
    model_key = ret_dict['model']

    frozen_model_path = "build/{}-{}/frozen_model".format(model_key, sequence)

    frozen_model_name = os.path.basename(get_newest(frozen_model_path))

    graph = load_graph(get_newest(frozen_model_path))

    test_path_list = read_test_data("dataset/test_dataset.txt")

    test_batch_generator = test_generator(test_path_list, batch_size, num_class, input_shape, resize_shape, \
                                          crop_x_range, crop_y_range, False, False, 0)

    test_object = test_all(graph, model_key, frozen_model_path, frozen_model_name, num_class, sequence)
    test_object.testing(keep_prob, test_batch_generator, config, True)
else:
    print('Sorry,{} isn’t a valid option'.format(sys.argv[1]))
    exit()

