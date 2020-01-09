import os
import time
import preprocess
import tensorflow as tf
from model import get_input_output_ckpt,unet
from util import one_hot,dice,iflarger,ifsmaller,frozen_graph,load_graph,get_newest,restore_from_pb
from preprocess import read_train_data,train_batch
from sklearn.model_selection import train_test_split

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

# hyper parameters
batch_size = 4
max_epoches = 200
rate = 0.0001
input_shape = (256,256)
num_class = 7
last = True            # last为False，那么pattern就失去作用了，因为一切都将重新开始
start_epoch = 35
pattern = "pb"

root_path = preprocess.root_path
task_list = preprocess.task_list
train_path = os.path.join(root_path,"train",task_list[2])
train_list = os.listdir(train_path)

start = time.time()
data,mask = read_train_data(train_path,train_list,input_shape)
end = time.time()
data_train,data_valid,mask_train,mask_valid = train_test_split(data,mask,test_size=0.1,shuffle=True)
print("spend time:%.2fs\ndata_train_shape:{} mask_train_shape:{}\ndata_valid_shape:{}\
 mask_valid_shape:{}".format(data_train.shape,mask_train.shape,data_valid.shape,mask_valid.shape)%(end-start))

one_epoch_steps = data.shape[0]//batch_size

# 1~24 epoch 使用了随机水平竖直翻转、15°旋转，此时测试结果会有左右肺互相混淆的情况，这是因为左右肺的灰度太过于相似，并且小器官分割效果比较差
# 25 epoch 开始使用10°旋转，并禁用翻转
train_batch_object, valid_batch_object = train_batch(data_train, mask_train, False, 10, num_class),\
                                         train_batch(data_valid, mask_valid, False, 10, num_class)

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
            x,y_hat = get_input_output_ckpt(unet,input_shape,num_class)
            y = tf.placeholder(tf.float32,[None,256,256,num_class],name="label")

            lr = tf.Variable(rate,name='learning_rate')
            decay_ops = tf.assign(lr,lr/2,name='learning_rate_decay')

            y_softmax = tf.get_default_graph().get_tensor_by_name("softmax_y:0")
            y_result = tf.get_default_graph().get_tensor_by_name("segementation_result:0")

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=y_hat),name="loss")
            optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
            with tf.name_scope("dice"):
                dice_index = dice(y_softmax,y)
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
            x = meta_graph.get_tensor_by_name("input:0")
            y = meta_graph.get_tensor_by_name("label:0")
            y_softmax = meta_graph.get_tensor_by_name("softmax_y:0")
            y_result = meta_graph.get_tensor_by_name("segementation_result:0")
            loss = meta_graph.get_tensor_by_name('loss:0')
            lr = meta_graph.get_tensor_by_name('learning_rate:0')
            decay_ops = meta_graph.get_tensor_by_name('learning_rate_decay:0')
            optimizer = meta_graph.get_operation_by_name("Adam")
            dice_index = meta_graph.get_tensor_by_name("dice/truediv:0")
            graph = meta_graph

with graph.as_default():
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session(config=config,graph=graph)
    sess.run(init)
    if(last==True):
        if(pattern=="ckpt"):
            saver.restore(sess,"ckpt/latest_model")
        else:
            pb_name = get_newest("frozen_model")
            print("{},the latest frozen graph is loaded...".format(pb_name))
            frozen_graph = load_graph(pb_name)
            sess = restore_from_pb(sess, frozen_graph, meta_graph)

    valid_log = {"loss":{},"dice":{}}
    valid_log_epochwise = {"loss":[100000],"dice":[0]}
    saved_valid_log_epochwise = {"loss":[100000],"dice":[0]}
    learning_rate_descent_flag = 0
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
        for j in range(one_epoch_steps):
            # one step
            # get one batch data and label
            train_batch_x,train_batch_y = train_batch_object.get_batch(batch_size)
            _ = sess.run(optimizer,feed_dict={x:train_batch_x,y:train_batch_y})
            if((j+1)%20==0):
                valid_batch_x,valid_batch_y = valid_batch_object.get_batch(batch_size)
                dic,los = sess.run([dice_index,loss],feed_dict={x:valid_batch_x,y:valid_batch_y})
                valid_log["loss"][i].append(los)
                valid_log["dice"][i].append(dic)
                one_epoch_avg_loss += los/(one_epoch_steps//20)
                one_epoch_avg_dice += dic/(one_epoch_steps//20)
                show_string = "epoch:{} steps:{} valid_loss:{} valid_dice:{}".format(i+1,j+1,los,dic)
                print(show_string)
                temp.write(show_string+'\n')
        
        show_string = "=======================================================\n\
    epoch_end: epoch:{} epoch_avg_loss:{} epoch_avg_dice:{}\n".format(i+1,one_epoch_avg_loss,one_epoch_avg_dice)

        if(iflarger(valid_log_epochwise["dice"],one_epoch_avg_dice)):
            learning_rate_descent_flag += 1
        
        if(learning_rate_descent_flag == 3):
            rate_once = rate
            _,rate = sess.run([decay_ops,lr])
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