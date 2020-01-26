import os
import time
import process
import numpy as np
import tensorflow as tf

from model import get_input_output_ckpt,unet,unet_SE
from util import one_hot,tf_dice,iflarger,ifsmaller,frozen_graph,load_graph,\
                 get_newest,restore_from_pb,restore_part_from_pb,weight_loss,tuple_string_to_tuple
from process import read_train_data, train_batch,OAR_epoch_read,GTV_epoch_read,config_dict
from sklearn.model_selection import train_test_split

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

OAR_OR_GTV = 'OAR' # OAR:task1 or task3 GTV:task2 or task4
task_number = 'task1'

model_dict = {
    'unet' : unet,
    'unet-CAB' : unet_SE
}

config_dict_lite = config_dict[OAR_OR_GTV][task_number]
root_path = config_dict['root_path']
task_name = config_dict_lite['task_name']
train_path = os.path.join(root_path,"train",task_name)
train_list = os.listdir(train_path)

# hyper parameters
batch_size = 4
max_epoches = 200
rate = 0.00001
input_shape = tuple_string_to_tuple(config_dict_lite['input_shape'])
num_class = config_dict_lite['obj_num']+1
last = True             # If choosing to train by old weights.
start_epoch = 1
pattern = "ckpt"          # the mode how to load the weights.
crop_x_range = tuple_string_to_tuple(config_dict_lite['crop_x_range'])
crop_y_range = tuple_string_to_tuple(config_dict_lite['crop_y_range'])
model_select = "unet" # "unet" or "unet-CAB"

ckpt_path = "ckpt/{}/{}".format(task_name,model_select)
frozen_model_path = "frozen_model/{}/{}".format(task_name,model_select)

start = time.time()
data,mask = read_train_data(train_path,train_list,input_shape,crop_x_range,crop_y_range)
end = time.time()
print("read ALL...\ndata length:{} mask length:{}".format(len(data),len(mask)))

if not os.path.exists("train_valid.log"):
    temp = open("train_valid.log","w")
else:
    temp = open("train_valid.log","a")

graph = tf.Graph()
if(pattern != "ckpt" and pattern != "pb"):
    print("The pattern must be ckpt or pb.")
    exit()
else:
    if(pattern == "ckpt" or last == False):
        with graph.as_default():
            # weight = tf.constant(weight)
            x,y_hat = get_input_output_ckpt(model_dict[model_select],num_class)
            y = tf.placeholder(tf.float32,[None, None, None, num_class],name="input_y")
            lr_init = tf.placeholder(tf.float32,name='input_lr')

            lr = tf.Variable(rate,name='learning_rate')
            init_ops = tf.assign(lr,lr_init,name='initial_lr')
            
            decay_ops = tf.assign(lr,lr/2,name='learning_rate_decay')

            y_softmax = tf.get_default_graph().get_tensor_by_name("softmax_y:0")
            y_result = tf.get_default_graph().get_tensor_by_name("segementation_result:0")

            dice_index = tf_dice(y_softmax,y)
            # 明早起来看效果，而后换新损失函数，给通道给了个先验加权
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=y_hat),name="loss")
            optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
            dice_index_indentity = tf.identity(dice_index,name="dice")
    else:
        if(len([x for x in os.listdir("frozen_model") if(os.path.splitext(x) == ".pb")])):
            print("sorry,there is not pb file.")
            exit()
        else:
            # 本来frozen_model一般用来进行测试，但是可以强行将已经被固定为常量的变量逆转化为变量。
            # frozen_model与checkpoint_model除了常量和变量的op在计算图上有区别之外其他都是一样的。
            # 一个常量仅有read一个op，而变量却有read、initial、assign三个op
            # 主要是卷积层和batch_norm层，卷积层主要有卷积核和偏置两个变量，而batch_norm主要有均值、方差、映射斜率gama和映射偏置beta
            # AdamOptimizer这一操作会直接将之前所有变量根据链式法则都创建一个梯度，无论是卷积、batch_norm还是激活函数都有梯度
            # 从pb文件中加载获得常量之后，然后根据常量在新图中再构建一次，将这些常量的值赋给元图
            saver = tf.train.import_meta_graph('{}/best_model.meta'.format(ckpt_path))
            meta_graph = tf.get_default_graph()
            x = meta_graph.get_tensor_by_name("input_x:0")
            y = meta_graph.get_tensor_by_name("input_y:0")
            y_softmax = meta_graph.get_tensor_by_name("softmax_y:0")
            y_result = meta_graph.get_tensor_by_name("segementation_result:0")
            loss = meta_graph.get_tensor_by_name('loss:0')
            lr = meta_graph.get_tensor_by_name('learning_rate:0')
            init_ops = meta_graph.get_operation_by_name('initial_lr')
            decay_ops = meta_graph.get_operation_by_name('learning_rate_decay')
            optimizer = meta_graph.get_operation_by_name("Adam")
            dice_index = meta_graph.get_tensor_by_name("dice:0")
            graph = meta_graph

with graph.as_default():
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session(config=config,graph=graph)
    sess.run(init)
    if(last==True):
        if(pattern=="ckpt"):
            try:
                saver.restore(sess,"{}/best_model".format(ckpt_path))
                print("The latest checkpoint model is loaded...")
            except:
                sess = restore_from_pb(sess, load_graph(get_newest(frozen_model_path)), graph)
        else:
            pb_name = get_newest(frozen_model_path)
            print("{},the latest frozen graph is loaded...".format(pb_name))
            pb_graph = load_graph(pb_name)
            sess = restore_from_pb(sess, pb_graph, meta_graph)

    valid_log = {"loss":{},"dice":{}}
    valid_log_epochwise = {"loss":[100000],"dice":[0]}
    saved_valid_log_epochwise = {"loss":[100000],"dice":[0]}
    learning_rate_descent_flag = 0
    sess.run(init_ops,feed_dict={"input_lr:0":rate})
    if(not os.path.exists(ckpt_path)):
        os.makedirs(ckpt_path)
    if(not os.path.exists(frozen_model_path)):
        os.makedirs(frozen_model_path)
    for i in range(start_epoch-1,max_epoches):
        # one epoch
        valid_log["loss"][i] = []
        valid_log["dice"][i] = []
        temp = open("train_valid.log","a")
        one_epoch_avg_dice = 0
        one_epoch_avg_loss = 0

        # 每个epoch重新初始化一次训练数据集
        if(OAR_OR_GTV == "OAR"):
            get_rate = tuple_string_to_tuple(config_dict_lite['rate'])
            start = time.time()
            data_epoch,mask_epoch = OAR_epoch_read(data,mask,get_rate)
            end = time.time()
        elif(OAR_OR_GTV == "GTV"):
            get_positive_rate = tuple_string_to_tuple(config_dict_lite['positive_get_rate'])
            get_negetive_rate = tuple_string_to_tuple(config_dict_lite['negetive_get_rate'])
            start = time.time()
            data_epoch,mask_epoch = GTV_epoch_read(data,mask,get_positive_rate,get_negetive_rate)
            end = time.time()
        else:
            print("bone!")
            exit()
        data_epoch_train,data_epoch_valid,mask_epoch_train,mask_epoch_valid = \
        train_test_split(data_epoch,mask_epoch,test_size=0.2,shuffle=True)
        train_batch_object, valid_batch_object = train_batch(data_epoch_train, mask_epoch_train, False, False, 0, num_class),\
                                                 train_batch(data_epoch_valid, mask_epoch_valid, False, False, 0, num_class)
        one_epoch_steps = data_epoch_train.shape[0]//batch_size
        show_string = "\
epoch dataset initial spend time:%.2fs \
 epoch steps:{}\n \
 data_train_shape:{} mask_train_shape:{}\n \
 data_valid_shape:{} mask_valid_shape:{}".format(one_epoch_steps, \
 data_epoch_train.shape,mask_epoch_train.shape, \
 data_epoch_valid.shape,mask_epoch_valid.shape)%(end-start)
        print(show_string)
        temp.write(show_string+'\n')
        for j in range(one_epoch_steps):
            # one step
            # get one batch data and label
            train_batch_x,train_batch_y = train_batch_object.get_batch(batch_size)
            _ = sess.run(optimizer,feed_dict={x:train_batch_x,y:train_batch_y})
            if((j+1)%5==0):
                valid_batch_x,valid_batch_y = valid_batch_object.get_batch(batch_size)
                dic,los,rate = sess.run([dice_index,loss,lr],feed_dict={x:valid_batch_x,y:valid_batch_y})
                valid_log["loss"][i].append(los)
                valid_log["dice"][i].append(dic)
                one_epoch_avg_loss += los/(one_epoch_steps//5)
                one_epoch_avg_dice += dic/(one_epoch_steps//5)
                show_string = "epoch:{} steps:{} valid_loss:{} valid_dice:{} learning_rate:{}".format(i+1,j+1,los,dic,rate)
                print(show_string)
                temp.write(show_string+'\n')
        show_string = "=======================================================\n \
epoch_end: epoch:{} epoch_avg_loss:{} epoch_avg_dice:{}\n".format(i+1,one_epoch_avg_loss,one_epoch_avg_dice)

        if(not iflarger(saved_valid_log_epochwise["dice"],one_epoch_avg_dice)):
            learning_rate_descent_flag += 1
        
        show_string += "learning_rate_descent_flag:{}\n".format(learning_rate_descent_flag)

        if(learning_rate_descent_flag == 3):
            rate_once = rate
            _ = sess.run(decay_ops)
            rate = sess.run(lr)
            show_string += "learning rate decay from {} to {}\n".format(rate_once,rate)
            learning_rate_descent_flag = 0
        
        if(iflarger(saved_valid_log_epochwise["dice"],one_epoch_avg_dice)):
            show_string += "ckpt_model_save because of {}<={}\n".format(saved_valid_log_epochwise["dice"][-1],one_epoch_avg_dice)
            saver.save(sess, "{}/best_model".format(ckpt_path))
            pb_name = "{}/{}_%.3f.pb".format(frozen_model_path,i+1)%(one_epoch_avg_dice)
            show_string += "frozen_model_save {}\n".format(pb_name)
            show_string += frozen_graph(sess,pb_name)
            saved_valid_log_epochwise['dice'].append(one_epoch_avg_dice)
            learning_rate_descent_flag = 0

        if(ifsmaller(saved_valid_log_epochwise["loss"],one_epoch_avg_loss)):
            saved_valid_log_epochwise['loss'].append(one_epoch_avg_loss)

        valid_log_epochwise["loss"].append(one_epoch_avg_loss)
        valid_log_epochwise["dice"].append(one_epoch_avg_dice)
        show_string += "=======================================================\n"

        print(show_string)
        temp.write(show_string)
        temp.close()
    saver.save(sess, "{}/best_model".format(ckpt_path))
    frozen_graph(sess,"{}/last.pb".format(frozen_model_path))
    sess.close()