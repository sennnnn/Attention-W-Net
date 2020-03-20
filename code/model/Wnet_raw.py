
from .model_base import *

def Wnet_onestage(input, num_class, initial_channel=64, keep_prob=0.1):
    
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
    input = tf.nn.dropout(input, keep_prob=keep_prob)
    input = CBR(input, c)

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

    return input

def Wnet_twostage(input, num_class, initial_channel=64, keep_prob=0.1):

    c = initial_channel

    input = Wnet_onestage(input, num_class, c, keep_prob)

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
    input = tf.nn.dropout(input, keep_prob=keep_prob)
    input = CBR(input, c)

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

def Wnet(input, num_class, initial_channel=64, keep_prob=0.1):

    out = Wnet_twostage(input, num_class, initial_channel, keep_prob)

    return out

if __name__ == "__main__":
    pass