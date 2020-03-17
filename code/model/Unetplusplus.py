from .model_base import *

def meta_block(input, filters):
    input = CBR(input, filters)
    input = CBR(input, filters)

    return input

def Unetplusplus(input, num_class, initial_channel=64, keep_prob=0.1):
    """
    Unet++ network architecture.
    """
    c = initial_channel;x0_0 = input
    c1 = c;c2 = c*2;c3 = c*4;c4 = c*8;c5 = c*16
    x0_0 = meta_block(x0_0, c1)
    x1_0 = CBR(x0_0, c1, strides=2);x1_0 = meta_block(x1_0, c2)
    x0_1 = meta_block(tf.concat([x0_0, upsampling(x1_0, c2)], axis=-1), c1)

    x2_0 = CBR(x1_0, c2, strides=2);x2_0 = meta_block(x2_0, c3)
    x1_1 = meta_block(tf.concat([x1_0, upsampling(x2_0, c3)], axis=-1), c2)
    x0_2 = meta_block(tf.concat([x0_0, x0_1, upsampling(x1_1, c2)], axis=-1), c1)

    x3_0 = CBR(x2_0, c3, strides=2);x3_0 = meta_block(x3_0, c4)
    x2_1 = meta_block(tf.concat([x2_0, upsampling(x3_0, c4)], axis=-1), c3)
    x1_2 = meta_block(tf.concat([x1_0, x1_1, upsampling(x2_1, c3)], axis=-1), c2)
    x0_3 = meta_block(tf.concat([x0_0, x0_1, x0_2, upsampling(x1_2, c2)], axis=-1), c1)

    x4_0 = CBR(x3_0, c4, strides=2);x4_0 = meta_block(x4_0, c5)
    x3_1 = meta_block(tf.concat([x3_0, upsampling(x4_0, c5)], axis=-1), c4)
    x2_2 = meta_block(tf.concat([x2_0, x2_1, upsampling(x3_1, c4)], axis=-1), c3)
    x1_3 = meta_block(tf.concat([x1_0, x1_1, x1_2, upsampling(x2_2, c3)], axis=-1), c2)
    x0_4 = meta_block(tf.concat([x0_0, x0_1, x0_2, x0_3, upsampling(x1_3, c2)], axis=-1), c1)

    out = CBR(x0_4, num_class, kernel_size=1)

    return out
