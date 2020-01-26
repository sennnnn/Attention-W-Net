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
    # General residual block
    shortcut = input
    input = DBR(input, filters)
    input = DBR(input, filters)
    return input + shortcut

def artous_conv(input, filters, rate):
    # Artous/Dilated Convolution，Only defined in tf,nn module.
    origin_channels = input.get_shape().as_list()
    weights = tf.Variable(tf.random_normal(shape=[3,3,origin_channels,filters]))
    input = tf.nn.atrous_conv2d(input, weights, rate, "valid")
    return input

def bottle_neck_block(input, filters):
    # It can let your network deeper with less parameters.
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
    # Up-sampling Layer,implemented by transpose convolution.
    input = layers.conv2d_transpose(input,filters,kernel_size,strides,padding='same')
    input = layers.batch_normalization(input,momentum=DECAY_BATCH_NORM,epsilon=EPSILON)
    return input

def unet(input,num_class):
    # Baseline Unet constructure.
## Encoder ##
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
    # To avoid over-fitting.
    input = tf.nn.dropout(input, 0.1)
    input = DBR(input,1024)

## Decoder ##
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
## ##
    input = DBR(input,num_class,kernel_size=1)
    return input

def unet_SE(input,num_class):
    # Attention mechanism block will be useful to face multiple segementation object.
## Encoder ##
    input = DBR(input,64)
    input = channel_attention_block(input)
    input = DBR(input,64)
    fus1  = input
    input = DBR(input,64,2)

    input = DBR(input,128)
    input = channel_attention_block(input)
    input = DBR(input,128)
    fus2  = input
    input = DBR(input,128,2)

    input = DBR(input,256)
    input = channel_attention_block(input)
    input = DBR(input,256)
    fus3  = input
    input = DBR(input,256,2)

    input = DBR(input,512)
    input = channel_attention_block(input)
    input = DBR(input,512)
    fus4  = input
    input = DBR(input,512,2)
## ##

    input = DBR(input,1024)
    input = channel_attention_block(input)
    input = tf.nn.dropout(input, 0.1)
    input = DBR(input,1024)
    
## Decoder ##
    input = upsampling(input, 1024)
    input = tf.concat([fus4,input],axis=-1)
    input = DBR(input,512)
    input = channel_attention_block(input)
    input = DBR(input,512)

    input = upsampling(input, 512)
    input = tf.concat([fus3,input],axis=-1)
    input = DBR(input,256)
    input = channel_attention_block(input)
    input = DBR(input,256)

    input = upsampling(input, 256)
    input = tf.concat([fus2,input],axis=-1)
    input = DBR(input,128)
    input = channel_attention_block(input)
    input = DBR(input,128)

    input = upsampling(input, 128)
    input = tf.concat([fus1,input],axis=-1)
    input = DBR(input,64)
    input = channel_attention_block(input)
    input = DBR(input,64)
## ##
    input = DBR(input,num_class,kernel_size=1)
    return input

def output_layer(input,thresh):
    """
    Args:
    input:the softmax output of the net.
    thresh:the threshould to binarilize the input.

    Return:
    out:Binary predict.
    """
    out = tf.cast((input > thresh),tf.uint8)
    return out
    

def get_input_output_ckpt(unet,num_class):
    x = tf.placeholder(tf.float32, [None, None, None, 1],name='input_x')
    with tf.variable_scope('unet'):
        y = unet(x,num_class)
    y_softmax = tf.nn.softmax(y,name='softmax_y')
    y_result  = tf.argmax(y_softmax,axis=-1,name='segementation_result')
    return x,y


if __name__ == "__main__":
    """
    yeah~
    """