
from .model_base import *

def attention_gate_block(input_g, input_l, f_in):
    input_g = CB(input_g, f_in)
    out = input_l
    input_l = CB(input_l, f_in)
    fuse = tf.nn.relu(input_g+input_l)
    fuse = CB(fuse, 1, kernel_size=1)
    fuse = tf.nn.sigmoid(fuse)

    return out*fuse

def net(input, num_class, keep_prob=0.1, initial_channel=64):
    c = initial_channel

    fuse_list = []

    for _ in range(4):
        input = CBR(input, c)
        input = CBR(input, c)
        fuse_list.append(input)
        input = CBR(input, c, strides=2)
        c = c*2

    input = CBR(input, c)
    input = tf.nn.dropout(input, keep_prob)
    input = CBR(input, c)

    for index in range(4):
        c = c//2
        input = upsampling(input, c)
        fuse = attention_gate_block(input, fuse_list[(-1-index)], c//2)
        input = tf.concat([fuse, input], axis=-1)
        input = CBR(input, c)
        input = CBR(input, c)

    input = CBR(input, num_class, kernel_size=1)

    return input