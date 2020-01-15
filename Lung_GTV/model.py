import tensorflow as tf
import tensorflow.layers as layers
import numpy as np

# layer parameters
DECAY_BATCH_NORM = 0.9
EPSILON = 1E-05
LEAKY_RELU = 0.1

def DBR(input, filters, strides=1, kernel_size=3):
    input = layers.conv2d(input, filters, kernel_size, use_bias=True ,strides=strides, padding='same', \
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
    input = layers.batch_normalization(input,momentum=DECAY_BATCH_NORM,epsilon=EPSILON)
    input = tf.nn.leaky_relu(input,alpha=LEAKY_RELU,name='ac')
    return input

def res_block(input, filters):
    # 普通的残差块
    shortcut = input
    input = DBR(input, filters)
    input = DBR(input, filters)
    return input + shortcut

def artous_conv(input, filters, rate):
    # 空洞卷积，只在tf,nn模块中有定义
    origin_channels = input.get_shape().as_list()
    weights = tf.Variable(tf.random_normal(shape=[3,3,origin_channels,filters]))
    input = tf.nn.atrous_conv2d(input, weights, rate, "valid")
    return input

def bottle_neck_block(input, filters):
    # 在很大层次时需要这样做，但是也许不那么需要
    shortcut = input
    input = DBR(input, filters//4, 1)
    input = DBR(input, filters//4, 3)
    input = DBR(input, filters, 1)
    return input + shortcut

def channel_attention_block(input):
    with tf.variable_scope("channel"):
        norm = tf.random_uniform([input.get_shape().as_list()[-1]])
        va1 = tf.Variable(norm)
        input = input*va1
        return input

def upsampling(input,filters,kernel_size=3,strides=2):
    # 上采样层，用反卷积实现
    input = layers.conv2d_transpose(input,filters,kernel_size,strides,padding='same')
    input = layers.batch_normalization(input,momentum=DECAY_BATCH_NORM,epsilon=EPSILON)
    return input

def unet(input,num_class):

## 编码器 ##
    input = DBR(input,64)
    input = DBR(input,64)
    fus1  = input
    input = DBR(input,64,2)

    input = DBR(input,128)
    input = DBR(input,128)
    fus2  = input
    input = DBR(input,128,2)

    input = DBR(input,256)
    input = DBR(input,256)
    fus3  = input
    input = DBR(input,256,2)

    input = DBR(input,512)
    input = DBR(input,512)
    fus4  = input
    input = DBR(input,512,2)
## ##

    input = DBR(input,1024)
    # 加入防止过拟合的dropout层
    input = channel_attention_block(input)
    input = tf.nn.dropout(input, 0.1)
    input = DBR(input,1024)

## 解码器 ##
    input = upsampling(input, 1024)
    input = tf.concat([fus4,input],axis=-1)
    input = DBR(input,512)
    input = DBR(input,512)

    input = upsampling(input, 512)
    input = tf.concat([fus3,input],axis=-1)
    input = DBR(input,256)
    input = DBR(input,256)

    input = upsampling(input, 256)
    input = tf.concat([fus2,input],axis=-1)
    input = DBR(input,128)
    input = DBR(input,128)

    input = upsampling(input, 128)
    input = tf.concat([fus1,input],axis=-1)
    input = DBR(input,64)
    input = DBR(input,64)
    # with tf.variable_scope("output_layer"):
    input = DBR(input,num_class,kernel_size=1)
    return input

def unet_SE(input,num_class):
    # 由于unet下采样太多次了，所以我感觉对于小器官也许不需要那么多次的下采样，可以引入注意力机制
## 编码器
    input = channel_attention_block(input)
    input = DBR(input,64)
    input = res_block(input,64)
    fus1 = input
    input = DBR(input,64,2)

    input = DBR(input,128)
    input = res_block(input,128)
    fus2 = input
    input = DBR(input,128,2)
##

    input = DBR(input,256)
    input = res_block(input,256)
    input = channel_attention_block(input)
    input = tf.nn.dropout(input,0.2)
    input = res_block(input,256)

## 解码器 ##
    input = upsampling(input,256)
    input = tf.concat([fus2,input],axis=-1)
    input = DBR(input,128)
    input = res_block(input,128)

    input = upsampling(input,128)
    input = tf.concat([fus1,input],axis=-1)
    input = DBR(input,64)
    input = res_block(input,64)
    input = DBR(input,num_class,kernel_size=1)
    input = channel_attention_block(input)
##
    return input

def get_input_output_ckpt(unet,num_class):
    x = tf.placeholder(tf.float32, [None, None, None, 1],name='input_x')
    with tf.variable_scope('unet'):
        y = unet(x,num_class)
    y_softmax = tf.nn.softmax(y,name='softmax_y')
    y_result  = tf.argmax(y_softmax,axis=-1,name='segementation_result')
    return x,y

def get_input_output_pb(frozen_graph):
    with frozen_graph.as_default():
        x = graph.get_tensor_by_name("input:0")
        y_result = graph.get_tensor_by_name("segementation_result:0")
    return x,y_result


if __name__ == "__main__":
    # input = tf.placeholder(tf.float32, [None,256,256,1], name='input_x')
    out = get_input_output_ckpt(unet, (256,256), 7)
    # list = tf.global_variables()
    # [print(x) for x in tf.global_variables()]
    # print(len(tf.global_variables()))
    with tf.Session() as sess:
        graph = tf.get_default_graph()
        ops = graph.get_operations()
        temp = open('abc.txt','w')
        [temp.write(x.name+'\n') for x in ops]
        temp.close()