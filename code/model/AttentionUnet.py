
from .model_base import *

def AttentionBlock(input_g, input_l, f_in):
    input_g = CB(input_g, f_in)
    out = input_l
    input_l = CB(input_l, f_in)
    fuse = tf.nn.relu(input_g+input_l)
    fuse = CB(fuse, 1, kernel_size=1)
    fuse = tf.nn.sigmoid(fuse)

    return out*fuse

def AttentionUnet(input, num_class, keep_prob=0.1, initial_channel=64):
    c = initial_channel

    input = CBR(input, c)
    input = CBR(input, c)
    fuse1 = input
    input = CBR(input, c, strides=2)

    c = c*2
    input = CBR(input, c)
    input = CBR(input, c)
    fuse2 = input
    input = CBR(input, c, strides=2)

    c = c*2
    input = CBR(input, c)
    input = CBR(input, c)
    fuse3 = input
    input = CBR(input, c, strides=2)

    c = c*2
    input = CBR(input, c)
    input = CBR(input, c)
    fuse4 = input
    input = CBR(input, c, strides=2)

    c = c*2
    input = CBR(input, c)
    input = tf.nn.dropout(input, keep_prob)
    input = CBR(input, c)
    input = upsampling(input, c)

    c = c//2
    fuse4 = AttentionBlock(input, fuse4, c//2)
    input = tf.concat([fuse4, input], axis=-1)
    input = CBR(input, c)
    input = CBR(input, c)
    input = upsampling(input, c)

    c = c//2
    fuse3 = AttentionBlock(input, fuse3, c//2)
    input = tf.concat([fuse3, input], axis=-1)
    input = CBR(input, c)
    input = CBR(input, c)
    input = upsampling(input, c)

    c = c//2
    fuse2 = AttentionBlock(input, fuse2, c//2)
    input = tf.concat([fuse2, input], axis=-1)
    input = CBR(input, c)
    input = CBR(input, c)
    input = upsampling(input, c)

    c = c//2
    fuse1 = AttentionBlock(input, fuse1, c//2)
    input = tf.concat([fuse1, input], axis=-1)
    input = CBR(input, c)
    input = CBR(input, c)

    input = CBR(input, num_class, kernel_size=1)

    return input