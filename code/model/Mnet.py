
from model_base import *

def Mnet(input, num_class, initial_channel=64, keep_prob=0.1):
    c = initial_channel
    # start input and size
    raw_c = c
    left_leg = CBR(input, raw_c)
    # h,w = input.get_shape().as_list()[1:3]

    temp = x
    x = CBR(left_leg, c)
    x = tf.concat([x, temp], axis=-1)
    x = CBR(x, c)
    x = CBR(x, c, strides=2)

    c = c*2
    left_leg = CBR(left_leg, raw_c, strides=2)
    x = tf.concat([left_leg, x], axis=-1)
    temp = x
    x = CBR(input, c)
    x = tf.concat([x, temp], axis=-1)
    x = CBR(input, c)
    