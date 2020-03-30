
from .model_base import *

def one_stage(input, num_class, keep_prob=0.1, initial_channel=64):
    
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
    fuse5 = input

    c = c//2
    input = upsampling(input, c)
    input = tf.concat([fuse4, input], axis=-1)
    input = CBR(input, c)
    input = CBR(input, c)
    fuse4 = input

    c = c//2
    input = upsampling(input, c)
    input = tf.concat([fuse3, input], axis=-1)
    input = CBR(input, c)
    input = CBR(input, c)
    fuse3 = input

    c = c//2
    input = upsampling(input, c)
    input = tf.concat([fuse2, input], axis=-1)
    input = CBR(input, c)
    input = CBR(input, c)
    fuse2 = input

    c = c//2
    input = upsampling(input, c)
    input = tf.concat([fuse1, input], axis=-1)
    input = CBR(input, c)
    input = CBR(input, c)
    fuse1 = input

    return fuse5, fuse4, fuse3, fuse2, fuse1

def two_stage(input, num_class, keep_prob=0.1, initial_channel=64):

    c = initial_channel

    fuse5,fuse4,fuse3,fuse2,fuse1 = one_stage(input, num_class, keep_prob, c)

    input = tf.concat([fuse1, input], axis=-1)
    input = CBR(input, c)
    input = CBR(input, c)
    fuse1 = input
    input = CBR(input, c, strides=2)

    c = c*2
    input = tf.concat([fuse2, input], axis=-1)
    input = CBR(input, c)
    input = CBR(input, c)
    fuse2 = input
    input = CBR(input, c, strides=2)

    c = c*2
    input = tf.concat([fuse3, input], axis=-1)
    input = CBR(input, c)
    input = CBR(input, c)
    fuse3 = input
    input = CBR(input, c, strides=2)

    c = c*2
    input = tf.concat([fuse4, input], axis=-1)
    input = CBR(input, c)
    input = CBR(input, c)
    fuse4 = input
    input = CBR(input, c, strides=2)

    c = c*2
    input = tf.concat([fuse5, input], axis=-1)
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

def net(input, num_class, keep_prob=0.1, initial_channel=64):

    out = two_stage(input, num_class, keep_prob, initial_channel)

    return out

if __name__ == "__main__":
    pass