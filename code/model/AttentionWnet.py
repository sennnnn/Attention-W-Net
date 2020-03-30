
from .model_base import *

def AttentionBlock(input_g, input_l, f_in):
    input_g = CB(input_g, f_in)
    out = input_l
    input_l = CB(input_l, f_in)
    fuse = tf.nn.relu(input_g+input_l)
    fuse = CB(fuse, 1, kernel_size=1)
    fuse = tf.nn.sigmoid(fuse)

    return out*fuse

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
    fuse5 = input

    c = c//2
    input = upsampling(input, c)
    fuse4 = AttentionBlock(input, fuse4, c)
    input = tf.concat([fuse4, input], axis=-1)
    input = CBR(input, c)
    input = CBR(input, c)
    fuse4 = input

    c = c//2
    input = upsampling(input, c)
    fuse3 = AttentionBlock(input, fuse3, c)
    input = tf.concat([fuse3, input], axis=-1)
    input = CBR(input, c)
    input = CBR(input, c)
    fuse3 = input

    c = c//2
    input = upsampling(input, c)
    fuse2 = AttentionBlock(input, fuse2, c)
    input = tf.concat([fuse2, input], axis=-1)
    input = CBR(input, c)
    input = CBR(input, c)
    fuse2 = input

    c = c//2
    input = upsampling(input, c)
    fuse1 = AttentionBlock(input, fuse1, c)
    input = tf.concat([fuse1, input], axis=-1)
    input = CBR(input, c)
    input = CBR(input, c)
    fuse1 = input

    return fuse5, fuse4, fuse3, fuse2, fuse1

def Wnet_twostage(input, num_class, initial_channel=64, keep_prob=0.1):

    c = initial_channel

    fuse5,fuse4,fuse3,fuse2,fuse1 = Wnet_onestage(input, num_class, c, keep_prob)
    fuse1 = AttentionBlock(input, fuse1, c)
    input = tf.concat([fuse1, input], axis=-1)
    input = CBR(input, c)
    input = CBR(input, c)
    fuse1 = input
    input = CBR(input, c, strides=2)

    c = c*2
    fuse2 = AttentionBlock(input, fuse2, c)
    input = tf.concat([fuse2, input], axis=-1)
    input = CBR(input, c)
    input = CBR(input, c)
    fuse2 = input
    input = CBR(input, c, strides=2)

    c = c*2
    fuse3 = AttentionBlock(input, fuse3, c)
    input = tf.concat([fuse3, input], axis=-1)
    input = CBR(input, c)
    input = CBR(input, c)
    fuse3 = input
    input = CBR(input, c, strides=2)

    c = c*2
    fuse4 = AttentionBlock(input, fuse4, c)
    input = tf.concat([fuse4, input], axis=-1)
    input = CBR(input, c)
    input = CBR(input, c)
    fuse4 = input
    input = CBR(input, c, strides=2)

    c = c*2
    fuse5 = AttentionBlock(input, fuse5, c)
    input = tf.concat([fuse5, input], axis=-1)
    input = CBR(input, c)
    input = tf.nn.dropout(input, keep_prob=keep_prob)
    input = CBR(input, c)

    c = c//2
    input = upsampling(input, c)
    fuse4 = AttentionBlock(input, fuse4, c)
    input = tf.concat([fuse4, input], axis=-1)
    input = CBR(input, c)
    input = CBR(input, c)

    c = c//2
    input = upsampling(input, c)
    fuse3 = AttentionBlock(input, fuse3, c)
    input = tf.concat([fuse3, input], axis=-1)
    input = CBR(input, c)
    input = CBR(input, c)

    c = c//2
    input = upsampling(input, c)
    fuse2 = AttentionBlock(input, fuse2, c)
    input = tf.concat([fuse2, input], axis=-1)
    input = CBR(input, c)
    input = CBR(input, c)

    c = c//2
    input = upsampling(input, c)
    fuse1 = AttentionBlock(input, fuse1, c)
    input = tf.concat([fuse1, input], axis=-1)
    input = CBR(input, c)
    input = CBR(input, c)

    input = CBR(input, num_class, kernel_size=1)

    return input

def AttentionWnet(input, num_class, initial_channel=64, keep_prob=0.1):

    out = Wnet_twostage(input, num_class, initial_channel, keep_prob)

    return out

if __name__ == "__main__":
    pass