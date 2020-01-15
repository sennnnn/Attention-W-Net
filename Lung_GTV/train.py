import os
import time
import process
import numpy as np
import tensorflow as tf

from model import get_input_output_ckpt,unet
from util import one_hot,tf_dice,iflarger,ifsmaller,frozen_graph,load_graph,\
                 get_newest,restore_from_pb,restore_part_from_pb,weight_loss
from process import read_train_data, train_batch, root_path, task_list,epoch_read
from sklearn.model_selection import train_test_split

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

# hyper parameters
batch_size = 4
max_epoches = 200
rate = 0.000001
input_shape = (256,256)
num_class = 2           # 背景和GTV
last = True             # last为False，那么pattern就失去作用了，因为一切都将重新开始
start_epoch = 84         # epoch 1~epoch 7的损失函数会有log0从而导致梯度爆炸
pattern = "ckpt"
# 先验结果
# weight = [0.0007440585488760174,0.999255941451124]
weight = [455/(455+4398),4398/(455+4398)]
# weight = [3/8,5/8]

train_path = os.path.join(root_path,"train",task_list[3])
train_list = os.listdir(train_path)

start = time.time()
data,mask = read_train_data(train_path,train_list,input_shape)
end = time.time()
print("read ALL...\ndata shape:{} mask shape:{}".format(data.shape,mask.shape))

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
            weight = tf.constant(weight)
            x,y_hat = get_input_output_ckpt(unet,num_class)
            y = tf.placeholder(tf.float32,[None, None, None, num_class],name="input_y")
            lr_init = tf.placeholder(tf.float32,name='input_lr')

            lr = tf.Variable(rate,name='learning_rate')
            init_ops = tf.assign(lr,lr_init,name='initial_lr')
            
            decay_ops = tf.assign(lr,lr/2,name='learning_rate_decay')

            y_softmax = tf.get_default_graph().get_tensor_by_name("softmax_y:0")
            y_result = tf.get_default_graph().get_tensor_by_name("segementation_result:0")
            
            dice_index = tf_dice(y_softmax,y)
            # 明早起来看效果，而后换新损失函数，给通道给了个先验加权
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=weight*y_hat),name="loss")
            # 加权损失函数再加上dice的影响让loss与训练更相关
            # loss = tf.reduce_mean(weight_loss(y,y_hat,weight),name='loss')
            # epoch 16开始换优化器了
            optimizer = tf.train.MomentumOptimizer(learning_rate=lr,momentum=0.99).minimize(loss)
        
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
            saver = tf.train.import_meta_graph('ckpt/latest_model.meta')
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
                saver.restore(sess,"ckpt/latest_model")
                print("The latest checkpoint model is loaded...")
            except:
                sess = restore_from_pb(sess, load_graph(get_newest("frozen_model")), graph)
        else:
            pb_name = get_newest("frozen_model")
            print("{},the latest frozen graph is loaded...".format(pb_name))
            pb_graph = load_graph(pb_name)
            sess = restore_from_pb(sess, pb_graph, meta_graph)
    else:
        sess = restore_part_from_pb(sess,load_graph("old/model_transfer_learning/77_0.925.pb"),graph)
    valid_log = {"loss":{},"dice":{}}
    valid_log_epochwise = {"loss":[100000],"dice":[0]}
    saved_valid_log_epochwise = {"loss":[100000],"dice":[0]}
    learning_rate_descent_flag = 0
    sess.run(init_ops,feed_dict={"input_lr:0":rate})
    if(not os.path.exists("ckpt")):
        os.mkdir("ckpt")
    if(not os.path.exists("frozen_model")):
        os.mkdir("frozen_model")
    for i in range(start_epoch-1,max_epoches):
        # one epoch
        valid_log["loss"][i] = []
        valid_log["dice"][i] = []
        temp = open("train_valid.log","a")
        one_epoch_avg_dice = 0
        one_epoch_avg_loss = 0

        # 每个epoch重新初始化一次训练数据集
        start = time.time()
        data_epoch,mask_epoch = epoch_read(data,mask)
        end = time.time()
        # 一阶段
        # 0~27 epoch没有使用数据增强，28~46 epoch开始使用数据增强，仅有5°旋转
        # 47 epoch开始使用10°旋转
        # epoch 76之后发现数据增强也无法解决问题，甚至还没有不加入数据增强更好，所以我决定用另一种方式来处理结果
        # 本来输出的应该是softmax处理之后的结果，我们将其视为概率，那么argmax输出的话，太过于极端，放弃argmax，当
        # 预测为肿瘤的概率达到一定概率而不一定要达到argmax的程度就输出结果。
        # 而后是重新开始训练了，采取小切片来进行勾画了，即裁剪后对。
        # 三阶段
        # 1~10 epoch使用数据增强，翻转、平移、旋转15° 11 epoch开始使用30°旋转
        data_epoch_train,data_epoch_valid,mask_epoch_train,mask_epoch_valid = \
        train_test_split(data_epoch,mask_epoch,test_size=0.2,shuffle=True)
        train_batch_object, valid_batch_object = train_batch(data_epoch_train, mask_epoch_train, True, True, 30, num_class),\
                                                 train_batch(data_epoch_valid, mask_epoch_valid, True, True, 30, num_class)
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
                show_string = "epoch:{} steps:{} valid_loss:{} valid_dice:{} learning_rate:{}".format(i+1,j+1,los,dic,rate) + '  ' + str(np.max(valid_batch_y))
                print(show_string)
                temp.write(show_string+'\n')
        show_string = "=======================================================\n \
epoch_end: epoch:{} epoch_avg_loss:{} epoch_avg_dice:{}\n".format(i+1,one_epoch_avg_loss,one_epoch_avg_dice)

        if(iflarger(valid_log_epochwise["dice"],one_epoch_avg_dice)):
            learning_rate_descent_flag += 1
        
        if(learning_rate_descent_flag == 7):
            rate_once = rate
            _ = sess.run(decay_ops)
            rate = sess.run(lr)
            show_string += "learning rate decay from {} to {}\n".format(rate_once,rate)
            learning_rate_descent_flag = 0
        
        if(iflarger(saved_valid_log_epochwise["dice"],one_epoch_avg_dice)):
            show_string += "ckpt_model_save because of {}<={}\n".format(saved_valid_log_epochwise["dice"][-1],one_epoch_avg_dice)
            saver.save(sess, "ckpt/latest_model")
            pb_name = "frozen_model/{}_%.3f.pb".format(i+1)%(one_epoch_avg_dice)
            show_string += "frozen_model_save {}\n".format(pb_name)
            show_string += frozen_graph(sess,pb_name)
            saved_valid_log_epochwise['dice'].append(one_epoch_avg_dice)

        if(ifsmaller(saved_valid_log_epochwise["loss"],one_epoch_avg_loss)):
            saved_valid_log_epochwise['loss'].append(one_epoch_avg_loss)

        valid_log_epochwise["loss"].append(one_epoch_avg_loss)
        valid_log_epochwise["dice"].append(one_epoch_avg_dice)
        show_string += "=======================================================\n"

        print(show_string)
        temp.write(show_string)
        temp.close()
    saver.save(sess, "ckpt/latest_model")
    frozen_graph(sess,"last.pb")
    sess.close()