
from .model_base import *

def DAC(input):
    c = input.get_shape().as_list()[-1]
    
    sub1 = ACBR(input, c, 1)
    sub1 = ACBR(sub1, c, 3)
    sub1 = ACBR(sub1, c, 5)
    sub1 = ACBR(sub1, c, 1, 1)

    sub2 = ACBR(input, c, 1)
    sub2 = ACBR(sub2, c, 3)
    sub2 = ACBR(sub2, c, 1, 1)

    sub3 = ACBR(input, c, 3)
    sub3 = ACBR(input, c, 1, 1)

    sub4 = ACBR(input, c, 1)

    return input+sub1+sub2+sub3+sub4

def RMC(input):
    
    sub1 = layers.max_pooling2d(input, 2, strides=2, padding='same');sub1 = upsampling(sub1, 1, kernel_size=1)
    sub2 = layers.max_pooling2d(input, 3, strides=2, padding='same');sub2 = upsampling(sub2, 1, kernel_size=1)
    sub3 = layers.max_pooling2d(input, 5, strides=2, padding='same');sub3 = upsampling(sub3, 1, kernel_size=1)
    sub4 = layers.max_pooling2d(input, 6, strides=2, padding='same');sub4 = upsampling(sub4, 1, kernel_size=1)

    out = tf.concat([sub4, sub3, sub2, sub1, input], axis=-1)

    return out

def net(input, num_class, keep_prob=0.1, initial_channel=64):
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
    input = DAC(input)
    input = tf.nn.dropout(input, keep_prob)
    input = RMC(input)

    c = c//2
    input = upsampling(input, c)
    input = tf.concat([fuse4, input], axis=-1)
    input = CBR(input, c)
    input = CBR(input, c)

    c = c//2
    input = upsampling(input, c)
    input = tf.concat([fuse3, input], axis=-1)
    input = CBR(input, c)
    input = CBR(input, c)

    c = c//2
    input = upsampling(input, c)
    input = tf.concat([fuse2, input], axis=-1)
    input = CBR(input, c)
    input = CBR(input, c)

    c = c//2
    input = upsampling(input, c)
    input = tf.concat([fuse1, input], axis=-1)
    input = CBR(input, c)
    input = CBR(input, c)

    input = CBR(input, num_class, kernel_size=1)

    return input