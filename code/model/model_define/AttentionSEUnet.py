
from .model_base import *

"""
The hybrid construction of the SE Unet and the Attention Unet
"""

def attention_gate_block(input_g, input_l, f_in):
    """
    The implementation of Attention Gate Block from AttentionUnet. 
    """
    input_g = CB(input_g, f_in)
    out = input_l
    input_l = CB(input_l, f_in)
    fuse = tf.nn.relu(input_g+input_l)
    fuse = CB(fuse, 1, kernel_size=1)
    fuse = tf.nn.sigmoid(fuse)

    return out*fuse

def channel_attention_block(input):
    """
    Squeeze and excitation block implementation.
    """
    c = input.get_shape().as_list()[-1]
    weight = tf.reduce_mean(input, [1, 2])
    weight = layers.dense(weight, c//2)
    weight = tf.nn.relu(weight, name='ac')
    weight = layers.dense(weight, c)
    weight = tf.nn.sigmoid(weight, name='ac')
    weight = tf.reshape(weight, [-1, 1, 1, c])
    input = input*weight
    input = layers.batch_normalization(input, momentum=DECAY_BATCH_NORM, epsilon=EPSILON)
    
    return input

def net(input, num_class, keep_prob=0.1, initial_channel=64):
    c = initial_channel

    fuse_list = []

    for _ in range(4):
        input = CBR(input, c)
        input = channel_attention_block(input)
        input = CBR(input, c)
        fuse_list.append(input)
        input = CBR(input, c, strides=2)
        c = c*2

    input = CBR(input, c)
    input = channel_attention_block(input)
    input = tf.nn.dropout(input, keep_prob)
    input = CBR(input, c)

    for index in range(4):
        c = c//2
        input = upsampling(input, c)
        fuse = attention_gate_block(input, fuse_list[(-1-index)], c//2)
        input = tf.concat([fuse, input], axis=-1)
        input = CBR(input, c)
        input = channel_attention_block(input)
        input = CBR(input, c)

    input = CBR(input, num_class, kernel_size=1)

    return input