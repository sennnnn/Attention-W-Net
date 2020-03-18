import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import *
from model.model_base import *
from model.loss_metric import *
from model.model_util import frozen_graph,restore_from_pb,load_graph
from util import iflarger,ifsmaller,get_newest,dict_save,dict_load

class train_all(object):
    model_dict = \
    {
        'unet' : Unet.unet,
        'unet-CAB' : Unet_SE.unet_SE,
        'r2Unet':R2Unet.r2Unet,
        'attentionUnet':AttentionUnet.AttentionUnet,
        'hybridUnet':HybridUnet.hybridAttentionUnet,
        'unetPlusPlus':Unetplusplus.Unetplusplus,
        'CEnet':CEnet.CEnet
    }
    def __init__(self, last, pattern, model_key, pb_path, ckpt_path, num_class, initial_channel, target):
        self.graph = tf.Graph()
        self.last_flag = last
        self.pattern = pattern
        self.pb_path = pb_path
        self.ckpt_path = ckpt_path
        self.model_key = model_key
        self.num_class = num_class
        self.initial_channel = initial_channel
        self.valid_log_metric_only_path = 'build/{}-{}/valid_metric_loss_only.log'.format(model_key, target)
        self.detail_log_path = "build/{}-{}/valid_detail.log".format(model_key, target)

    def _input_compose(self):
        x = tf.placeholder(tf.float32, [None, None, None, 1], name='data')
        self.input_collecting = x

        return self.input_collecting

    def _train_graph_compose(self):
        if(self.pattern != "ckpt" and self.pattern != "pb"):
            print("The pattern must be ckpt or pb.")
            exit()
        elif(self.pattern == "ckpt" or self.last_flag == 'False'):
            with self.graph.as_default() as g:
                # network input & output
                x = tf.placeholder(tf.float32, [None, None, None, 1], name='data')
                y = tf.placeholder(tf.float32, [None, None, None, self.num_class], name='label')
                keep_prob = tf.placeholder(tf.float32, name='dropout_rate')
                construct_network(self.model_dict[self.model_key], x, self.num_class, self.initial_channel, \
                                keep_prob)

                # learning rate relating things
                lr_input = tf.placeholder(tf.float32, name='lr_input')
                lr = tf.Variable(1., name='lr')

                lr_init_op = tf.assign(lr, lr_input, name='lr_initial_op')
                lr_decay_op = tf.assign(lr, lr/2, name='lr_decay_op')

                self.lr_relating = {'lr':lr, 'lr_initial_op':lr_init_op, 'lr_decay_op':lr_decay_op}

                # loss & metric & optimizer relating things.
                predict = g.get_tensor_by_name('predict:0')
                softmax = g.get_tensor_by_name('predict_softmax:0')
                argmax = g.get_tensor_by_name('predict_argmax:0')
                
                self.output_relating = {'predict_softmax':softmax, 'predict_argmax':argmax}

                dice = tf_dice(softmax, y)
                dice_identity = tf.identity(dice, name='dice')
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=predict))
                loss_identity = tf.identity(loss, name='loss')
                
                self.loss = loss
                self.metric = dice

                self.optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='optimizer').minimize(loss)
                
        else:
            if(len([x for x in os.listdir(self.pb_path) if(os.path.splitext(x) == ".pb")])):
                print("sorry,there is not pb file.")
                exit()
            else:
                with self.graph.as_default() as g:
                    saver = tf.train.import_meta_graph('{}/best_model.meta'.format(self.ckpt_path))

                    # learning rate relating things
                    lr = g.get_tensor_by_name('lr:0')
                    lr_init_op = g.get_operation_by_name('lr_initial_op')
                    lr_decay_op = g.get_operation_by_name('lr_decay_op')

                    self.lr_relating = {'lr':lr, 'lr_initial_op':lr_init_op, 'lr_decay_op':lr_decay_op}

                    # loss & metric & optimizer relating things.                
                    softmax = g.get_tensor_by_name('predict_softmax:0')
                    argmax = g.get_tensor_by_name('predict_argmax:0')

                    self.output_relating = {'predict_softmax':softmax, 'predict_argmax':argmax}

                    loss = g.get_tensor_by_name('loss:0')
                    dice = g.get_tensor_by_name("dice:0")
                    
                    self.loss = loss 
                    self.metric = dice

                    self.optimizer = g.get_operation_by_name('optimizer')

    def _restore(self, sess, saver, graph, pattern):
        if(pattern == "ckpt"):
            try:
                saver.restore(sess,"{}/best_model".format(self.ckpt_path))
                print("The latest checkpoint model is loaded...")
            except:
                sess = restore_from_pb(sess, load_graph(get_newest("{}".format(self.pb_path))), graph)
        else:
            pb_name = get_newest("{}".format(self.pb_path))
            print("{},the latest frozen graph is loaded...".format(pb_name))
            pb_graph = load_graph(pb_name)
            sess = restore_from_pb(sess, pb_graph, graph)

    def _log_write(self, show_string=None, end=False):
        if(show_string == None):
            if not os.path.exists(self.detail_log_path):
                self.log_file = open(self.detail_log_path, "w")
            else:
                self.log_file = open(self.detail_log_path, "a")
        else:
            print(show_string)
            self.log_file.write(show_string + '\n')
        
        if(end):
            self.log_file.close()

    def _log_dict_save(self, start=False):
        if(start):
            if(os.path.exists(self.valid_log_metric_only_path)):
                self.valid_log_dict = dict_load(self.valid_log_metric_only_path)
            else:
                self.valid_log_dict = {}
                # 用于记录训练过程，分为两种模式
                self.valid_log_dict['stepwise'] = {'loss':{}, 'metric':{}}
                self.valid_log_dict['epochwise'] = {'loss':[], 'metric':[]}
        else:
            dict_save(self.valid_log_dict, self.valid_log_metric_only_path)

    def _model_save(self, sess, saver, epoch, pattern, one_epoch_avg_metric):
        return_string = ''
        if(pattern == 'ckpt'):
            saver.save(sess, "{}/best_model".format(self.ckpt_path))
            pb_name = "{}/{}_%.3f.pb".format(self.pb_path, epoch)%(one_epoch_avg_metric)
            return_string += 'frozen_model_save {}\n'.format(pb_name)
            return_string += frozen_graph(sess, pb_name)
        elif(pattern == 'pb'):
            # 因为是从 frozen_model 中 restore 所以应该将 restore_from_pb 中多出来的那些 assign 操作去掉
            saver.save(sess, "{}/best_model".format(self.ckpt_path))
            pb_name = "{}/{}_%.3f.pb".format(self.pb_path, epoch)%(one_epoch_avg_metric)
            return_string += 'frozen_model_save {}\n'.format(pb_name)
            return_string += frozen_graph(sess, pb_name)
        else:
            print('pattern must be ckpt or pb!')
            exit()

        return return_string
    
    def training(self, learning_rate, max_epoches, one_epoch_steps, start_epoch,\
                 train_generator, valid_generator, decay_patientce, valid_step, \
                 keep_prob, tf_config):
        self._log_dict_save(start=True)
        self._train_graph_compose()
        with self.graph.as_default() as g:
            init = tf.global_variables_initializer()
            saver = tf.train.Saver(tf.global_variables(scope='network'))
            sess = tf.Session(config=tf_config, graph=g)
            sess.run(init)
            if(self.last_flag):
                self._restore(sess, saver, g, self.pattern)
            # 记录训练过程的字符串
            show_string = None
            # 用于比较是否进行学习率衰减的
            saved_valid_log_epochwise = {'loss':[100000], 'metric':[0]}
            # 学习率衰减标志位
            learning_rate_descent_flag = 0
            # 将学习率初始化
            sess.run(self.lr_relating['lr_initial_op'], feed_dict={"lr_input:0":learning_rate})
            # 开始训练前，检查一遍权重保存路径
            if(not os.path.exists(self.ckpt_path)):
                os.makedirs(self.ckpt_path, 0o777)
            if(not os.path.exists(self.pb_path)):
                os.makedirs(self.pb_path, 0o777)
            for i in range(start_epoch-1, max_epoches):
                # one epoch
                # 打开 log 文件
                self._log_write()
                # 不同 epoch 的训练中间结果分别保存
                self.valid_log_dict['stepwise']['loss'][i+1] = []
                self.valid_log_dict['stepwise']['metric'][i+1] = []
                # 用来学习率衰减的指标
                one_epoch_avg_loss = 0
                one_epoch_avg_metric = 0
                epochwise_train_generator = train_generator.epochwise_iter()
                epochwise_valid_generator = valid_generator.epochwise_iter()
                show_string = "epoch {}\ntrain dataset number:{} valid dataset number:{}"\
                              .format(i+1, train_generator.slice_count, valid_generator.slice_count)
                self._log_write(show_string=show_string)
                for j in range(one_epoch_steps):
                    # one step
                    data,label = next(epochwise_train_generator)
                    feed_dict = {'data:0': data, 'label:0': label, 'dropout_rate:0': keep_prob}
                    # 三个输入，拿 T1 的作为标签
                    _ = sess.run(self.optimizer, feed_dict=feed_dict)
                    if((j+1)%valid_step == 0):
                        data,label = next(epochwise_valid_generator)
                        feed_dict = {'data:0': data, 'label:0': label, 'dropout_rate:0': keep_prob}
                        los,met = sess.run([self.loss, self.metric], feed_dict=feed_dict)
                        # 保存每次 valid 的指标
                        self.valid_log_dict['stepwise']["loss"][i+1].append(los)
                        self.valid_log_dict['stepwise']["metric"][i+1].append(met)
                        one_epoch_avg_loss += los
                        one_epoch_avg_metric += met
                        show_string = "epoch:{} steps:{}/{} valid_loss:{} valid_dice:{} learning_rate:{}"\
                                      .format(i+1, j+1, one_epoch_steps, los, met, learning_rate)
                        self._log_write(show_string=show_string)

                one_epoch_avg_loss = one_epoch_avg_loss/(one_epoch_steps//valid_step)
                one_epoch_avg_metric = one_epoch_avg_metric/(one_epoch_steps//valid_step)
                show_string = "=======================================================\n\
epoch_end epoch:{} epoch_avg_loss:{} epoch_avg_metric:{}\n"\
                .format(i+1, one_epoch_avg_loss, one_epoch_avg_metric)

                # 以 metric 为基准，作为学习率衰减的参考指标
                if(not iflarger(saved_valid_log_epochwise["metric"], one_epoch_avg_metric)):
                    learning_rate_descent_flag += 1
                
                show_string += "learning_rate_descent_flag:{}\n".format(learning_rate_descent_flag)
                
                if(learning_rate_descent_flag == decay_patientce):
                    learning_rate_once = learning_rate
                    _ = sess.run(self.lr_relating['lr_decay_op'])
                    learning_rate = sess.run(self.lr_relating['lr'])
                    show_string += "learning rate decay from {} to {}\n".format(learning_rate_once, learning_rate)
                    learning_rate_descent_flag = 0

                if(iflarger(saved_valid_log_epochwise["metric"], one_epoch_avg_metric)):
                    show_string += "ckpt_model_save because of {}<={}\n"\
                                    .format(saved_valid_log_epochwise["metric"][-1], one_epoch_avg_metric)
                    show_string += self._model_save(sess, saver, i+1, self.pattern, one_epoch_avg_metric)
                    saved_valid_log_epochwise['metric'].append(one_epoch_avg_metric)
                    learning_rate_descent_flag = 0

                if(ifsmaller(saved_valid_log_epochwise["loss"], one_epoch_avg_loss)):
                    saved_valid_log_epochwise['loss'].append(one_epoch_avg_loss)

                # 保存每个 epoch 的平均指标
                self.valid_log_dict['epochwise']['loss'].append(one_epoch_avg_loss)
                self.valid_log_dict['epochwise']['metric'].append(one_epoch_avg_metric)
                
                # 保存 log_dict
                self._log_dict_save()

                show_string += "======================================================="
                
                self._log_write(show_string=show_string)
                self._log_write(end=True)
            saver.save(sess, "{}/best_model".format(self.ckpt_path))
            frozen_graph(sess,"{}/last.pb".format(self.pb_path))
            sess.close()