import os
import time
import random
import preprocess
import tensorflow as tf
import numpy as np
from preprocess import test_batch,read_test_data
from model import unet,get_input_output_ckpt
from util import dice
import matplotlib.pyplot as plt

root_path = preprocess.root_path
task_list = preprocess.task_list
test_path = os.path.join(root_path,"test",task_list[2])
test_list = os.listdir(test_path)

start = time.time()
data,mask = read_test_data(test_path,test_list)
end = time.time()
print("spend time:%.2fs\ndata_shape:{} mask_shape:{}".format(data.shape,mask.shape)%(end-start))

# batch_object = test_batch()

def test_one_patient(one_patient_data,one_patient_mask,batch_size):
    # hyper parameters
    input_shape = (224,224)
    num_class = 7

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.InteractiveSession(config=config)

    # batch对象只能走一个轮回就结束了，单个病人不能循环
    batch_object = test_batch(one_patient_data,one_patient_mask)

    x,y_hat = get_input_output_ckpt(unet,input_shape,num_class)
    y = tf.placeholder(tf.float32,[None,224,224,7],name="label")
    y_softmax = tf.get_default_graph().get_tensor_by_name("softmax_y:0")
    y_result = tf.get_default_graph().get_tensor_by_name("segementation_result:0")

    dice_index = dice(y_softmax,y)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session(config=config) as sess:
        sess.run(init)
        # 没有ckpt模型restore会失败，程序会退出，而之后也不会执行
        saver.restore(sess, "ckpt/latest_model")
        number = 0
        patient_mask_predict = []
        while(1):
            number += 1
            batch,flag = batch_object.get_batch(batch_size)
            print(batch,flag)
            if(not flag):
                break
            batch_test_x,batch_test_y = batch[0],batch[1]
            plt.subplot(121)
            plt.imshow(batch_test_x[2,:,:,0],cmap='gray')
            plt.axis('off')
            plt.subplot(122)
            plt.imshow(np.argmax(batch_test_y[2],axis=-1),cmap='gray')
            plt.axis('off')
            result,dic = sess.run([y_result,dice_index], feed_dict={x:batch_test_x,y:batch_test_y})
            temp = random.randint(0,3)
            plt.figure()
            plt.imshow(result[temp],cmap='gray')
            plt.axis('off')
            plt.title("dice:%.3f"%(dic))
            plt.show()
        print(number)
for one_patient_data,one_patient_mask in zip(data,mask):
    test_one_patient(one_patient_data,one_patient_mask,4)