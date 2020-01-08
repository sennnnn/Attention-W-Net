import tensorflow as tf
import tensorflow.layers as layers
import numpy as np

# layer parameters
DECAY_BATCH_NORM = 0.9
EPSILON = 1E-05
LEAKY_RELU = 0.1

def DBR(input, filters, strides=1, kernel_size=3):
    input = layers.conv2d(input, filters, kernel_size, use_bias=True ,strides=strides, padding='same')
    input = layers.batch_normalization(input,momentum=DECAY_BATCH_NORM,epsilon=EPSILON)
    input = tf.nn.leaky_relu(input,alpha=LEAKY_RELU,name='ac')
    return input

def res_block(input, filters):
    # 普通的残差块
    shortcut = input
    input = DBR(input, filters, 3)
    input = DBR(input, filters, 3)
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

def upsampling(input,filters,kernel_size=3,strides=2):
    # 上采样层，用反卷积实现
    input = layers.conv2d_transpose(input,filters,kernel_size,strides,padding='same')
    input = layers.batch_normalization(input,momentum=DECAY_BATCH_NORM,epsilon=EPSILON)
    return input

def unet(input,num_class):
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

    input = DBR(input,1024)
    input = DBR(input,1024)

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
    input = DBR(input,num_class,kernel_size=1)
    return input

def get_input_output_ckpt(unet,input_shape,num_class):
    x = tf.placeholder(tf.float32, [None, *input_shape, 1],name='input')
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
    input = tf.placeholder(tf.float32, [None,224,224,1], name='input')
    out = unet(input)
    list = tf.global_variables()
    # [print(x) for x in tf.global_variables()]
    # print(len(tf.global_variables()))
    # list = tf.global_variables()
    # op = tf.assign(list[1],np.zeros(3))
    with tf.Session() as sess:
        graph = tf.get_default_graph()
        print(list[0].graph)
        print(graph)
        abc = tf.Graph()
        print(abc)
        # print(sess.run(ops))