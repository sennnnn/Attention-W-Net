import os
import numpy as np
import tensorflow as tf

from util import average,saveAsNiiGz
from process import recover
from model.loss_metric import np_dice_index,np_dice_index_channel_wise

class test_all(object):
    def __init__(self, graph, model_key, pb_path, pb_name, num_class, target):
        self.graph = graph
        self.model_key = model_key
        self.pb_path = pb_path
        self.num_class = num_class
        self.target = target
        self.test_result_root_path = 'build/{}-{}/test_result/{}'.format(model_key, target, pb_name)
        if(not os.path.exists(self.test_result_root_path)):
            os.makedirs(self.test_result_root_path, 0x777)

    def _predict_argmax_recover_to_nii(self, predict_argmax_recover, save_dir_path, info_dict):
        if(not os.path.exists(save_dir_path)):
            os.makedirs(save_dir_path, 0x777)
        save_path = save_dir_path + '/predict.nii.gz'
        saveAsNiiGz(predict_argmax_recover, save_path, info_dict['spacing'], info_dict['origin'])

    def _test_graph_compose_and_restore(self):
        with self.graph.as_default() as g:
            # loss & metric & optimizer relating things.
            predict = g.get_tensor_by_name('predict:0')
            softmax = tf.nn.softmax(predict)
            argmax = tf.argmax(softmax, axis=-1)
        
            self.output_relating = {'predict_softmax':softmax, 'predict_argmax':argmax}

    def testing(self, keep_prob, test_generator, tf_config, ifprocess=True):
        crop_x_range = test_generator.crop_x_range
        crop_y_range = test_generator.crop_y_range
        resize_shape = test_generator.resize_shape
        patientwise_test_generator = test_generator.patientwise_iter()
        num_class = self.num_class
        self._calculate_and_save_metric(start=True)
        self._test_graph_compose_and_restore()
        with self.graph.as_default() as g:
            sess = tf.Session(graph=g, config=tf_config)
            for patient_name,nii_meta_info,single_patient_generator in patientwise_test_generator:
                # 单个病人
                patient_test_argmax = []
                patient_test_label = []
                for data,label in single_patient_generator:
                    feed_dict = {'data:0': data, 'dropout_rate:0': keep_prob}
                    argmax = sess.run(self.output_relating['predict_argmax'], feed_dict)
                    for i in range(argmax.shape[0]):
                        patient_test_argmax.append(argmax[i])
                        patient_test_label.append(label[i])
                patient_test_label = np.array(patient_test_label)
                patient_test_argmax_recover = recover(patient_test_argmax, patient_test_label.shape[:3], \
                                                      ifprocess, num_class, crop_x_range, crop_y_range, resize_shape)
                patient_test_argmax_recover = patient_test_argmax_recover.astype(np.uint8)
                test_result_save_dir_path = '{}/{}'.format(self.test_result_root_path, patient_name)
                self._predict_argmax_recover_to_nii(np.argmax(patient_test_argmax_recover, axis=-1), \
                                                    test_result_save_dir_path, nii_meta_info)

