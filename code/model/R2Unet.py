
from .model_base import *

def r_block(input, filters, t):
    for i in range(t):
        add = input
        input = CBR(add, filters)
        add = input + add

    return input

def rr_block(input, filters, t):
    input = CBR(input, filters, kernel_size=1)
    input = r_block(input, filters, t)
    input = r_block(input, filters, t)

    return input

def R2Unet(input, num_class, keep_prob=0.1, initial_channel=64, t=2):
    c = initial_channel

    input = rr_block(input, c, t)
    fus1 = input
    input = CBR(input, c, strides=2)

    c = c*2
    input = rr_block(input, c, t)
    fus2 = input
    input = CBR(input, c, strides=2)

    c = c*2
    input = rr_block(input, c, t)
    fus3 = input
    input = CBR(input, c, strides=2)

    c = c*2
    input = rr_block(input, c, t)
    fus4 = input
    input = CBR(input, c, strides=2)

    c = c*2
    input = tf.nn.dropout(input, keep_prob)
    input = rr_block(input, c, t)
    input = upsampling(input, c)

    c = c//2
    input = tf.concat([fus4, input], axis=-1)
    input = rr_block(input, c, t)
    input = upsampling(input, c)

    c = c//2
    input = tf.concat([fus3, input], axis=-1)
    input = rr_block(input, c, t)
    input = upsampling(input, c)

    c = c//2
    input = tf.concat([fus2, input], axis=-1)
    input = rr_block(input, c, t)
    input = upsampling(input, c)

    c = c//2
    input = tf.concat([fus1, input], axis=-1)
    input = rr_block(input, c, t)
    input = CBR(input, num_class, kernel_size=1)

    return input