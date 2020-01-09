import os
import time
import preprocess
import tensorflow as tf
from model import get_input_output_ckpt,unet
from util import one_hot,dice,iflarger,ifsmaller,frozen_graph
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
last = True
start_epoch = 31

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


x,y_hat = get_input_output_ckpt(unet,input_shape,num_class)
y = tf.placeholder(tf.float32,[None,256,256,num_class],name="label")

lr = tf.Variable(rate,name='learning_rate')
decay_ops = tf.assign(lr,lr/2)

y_softmax = tf.get_default_graph().get_tensor_by_name("softmax_y:0")
y_result = tf.get_default_graph().get_tensor_by_name("segementation_result:0")

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=y_hat),name="loss")
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

with tf.name_scope("dice"):
    dice_index = dice(y_softmax,y)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
if not os.path.exists("train_valid.log"):
    temp = open("train_valid.log","w")
else:
    temp = open("train_valid.log","a")

with tf.Session(config=config) as sess:
    sess.run(init)
    if(last==True):
        saver.restore(sess,"ckpt/latest_model")
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
            show_string += "learning rate decay from {} to {}\n".format(rate,rate/2)
            rate = rate/2
            sess.run(decay_ops)
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