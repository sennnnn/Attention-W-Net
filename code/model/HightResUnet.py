
from .model_base import *

def rac(input, channel, rate):
    """
    residual atrous convolution block.
    """
    raw = input
    input = ACBR(input, channel, rate)
    input = ACBR(input, channel, rate)
    out = input + raw

    return out

def HightResUnet(input, num_class, initial_channel=64, keep_prob=0.1):

    c = initial_channel

    input = CBR(input, c)
    input = rac(input, c, 1)
    input = rac(input, c, 1)
    fuse1 = input

    c = c*2
    input = CBR(input, c)
    input = rac(input, c, 2)
    input = rac(input, c, 2)
    fuse2 = input

    c = c*2
    input = CBR(input, c)
    input = rac(input, c, 4)
    input = tf.nn.dropout(input, keep_prob=keep_prob)
    input = rac(input, c, 4)

    c = c//2
    input = tf.concat([fuse2, input], axis=-1)
    input = CBR(input, c)
    input = rac(input, c, 2)
    input = rac(input, c, 2)

    c = c//2
    input = tf.concat([fuse1, input], axis=-1)
    input = CBR(input, c)
    input = rac(input, c, 1)
    input = rac(input, c, 1)

    input = CBR(input, num_class, kernel_size=1)

    return input
    