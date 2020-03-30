
from .model_base import *

def channel_attention_block(input):
    """
    SE block implementation.
    What's more,the operations in this function will be in the channel variable scope.
    Args:
        input:tensor that will be operated.
    Return:
        input:input * channel weight.
    """
    c = input.get_shape().as_list()[-1]
    weight = tf.reduce_mean(input, [1, 2])
    weight = layers.dense(weight, c//2)
    weight = tf.nn.relu(weight, name='ac')
    weight = layers.dense(weight, c)
    weight = tf.nn.sigmoid(weight, name='ac')
    weight = tf.reshape(weight, [-1, 1, 1, c])
    input = input*weight
    input = layers.batch_normalization(input, momentum=DECAY_BATCH_NORM, epsilon=EPSILON)
    
    return input

def Wnet_onestage(input, num_class, keep_prob=0.1, initial_channel=64):
    
    c = initial_channel
    input = CBR(input, c)
    input = channel_attention_block(input)
    input = CBR(input, c)
    fuse1 = input
    input = CBR(input, c, strides=2)

    c = c*2
    input = CBR(input, c)
    input = channel_attention_block(input)
    input = CBR(input, c)
    fuse2 = input
    input = CBR(input, c, strides=2)

    c = c*2
    input = CBR(input, c)
    input = channel_attention_block(input)
    input = CBR(input, c)
    fuse3 = input
    input = CBR(input, c, strides=2)

    c = c*2
    input = CBR(input, c)
    input = channel_attention_block(input)
    input = CBR(input, c)
    fuse4 = input
    input = CBR(input, c, strides=2)

    c = c*2
    input = CBR(input, c)
    input = channel_attention_block(input)
    input = tf.nn.dropout(input, keep_prob=keep_prob)
    input = CBR(input, c)
    fuse5 = input

    c = c//2
    input = upsampling(input, c)
    input = tf.concat([fuse4, input], axis=-1)
    input = CBR(input, c)
    input = channel_attention_block(input)
    input = CBR(input, c)
    fuse4 = input

    c = c//2
    input = upsampling(input, c)
    input = tf.concat([fuse3, input], axis=-1)
    input = CBR(input, c)
    input = channel_attention_block(input)
    input = CBR(input, c)
    fuse3 = input

    c = c//2
    input = upsampling(input, c)
    input = tf.concat([fuse2, input], axis=-1)
    input = CBR(input, c)
    input = channel_attention_block(input)
    input = CBR(input, c)
    fuse2 = input

    c = c//2
    input = upsampling(input, c)
    input = tf.concat([fuse1, input], axis=-1)
    input = CBR(input, c)
    input = channel_attention_block(input)
    input = CBR(input, c)
    fuse1 = input

    return fuse5, fuse4, fuse3, fuse2, fuse1

def two_stage(input, num_class, keep_prob=0.1, initial_channel=64):

    c = initial_channel

    fuse5,fuse4,fuse3,fuse2,fuse1 = one_stage(input, num_class, keep_prob, c)

    input = tf.concat([fuse1, input], axis=-1)
    input = CBR(input, c)
    input = channel_attention_block(input)
    input = CBR(input, c)
    fuse1 = input
    input = CBR(input, c, strides=2)

    c = c*2
    input = tf.concat([fuse2, input], axis=-1)
    input = CBR(input, c)
    input = channel_attention_block(input)
    input = CBR(input, c)
    fuse2 = input
    input = CBR(input, c, strides=2)

    c = c*2
    input = tf.concat([fuse3, input], axis=-1)
    input = CBR(input, c)
    input = channel_attention_block(input)
    input = CBR(input, c)
    fuse3 = input
    input = CBR(input, c, strides=2)

    c = c*2
    input = tf.concat([fuse4, input], axis=-1)
    input = CBR(input, c)
    input = channel_attention_block(input)
    input = CBR(input, c)
    fuse4 = input
    input = CBR(input, c, strides=2)

    c = c*2
    input = tf.concat([fuse5, input], axis=-1)
    input = CBR(input, c)
    input = channel_attention_block(input)
    input = tf.nn.dropout(input, keep_prob=keep_prob)
    input = CBR(input, c)

    c = c//2
    input = upsampling(input, c)
    input = tf.concat([fuse4, input], axis=-1)
    input = CBR(input, c)
    input = channel_attention_block(input)
    input = CBR(input, c)

    c = c//2
    input = upsampling(input, c)
    input = tf.concat([fuse3, input], axis=-1)
    input = CBR(input, c)
    input = channel_attention_block(input)
    input = CBR(input, c)

    c = c//2
    input = upsampling(input, c)
    input = tf.concat([fuse2, input], axis=-1)
    input = CBR(input, c)
    input = channel_attention_block(input)
    input = CBR(input, c)

    c = c//2
    input = upsampling(input, c)
    input = tf.concat([fuse1, input], axis=-1)
    input = CBR(input, c)
    input = channel_attention_block(input)
    input = CBR(input, c)

    input = CBR(input, num_class, kernel_size=1)

    return input

def net(input, num_class, keep_prob=0.1, initial_channel=64):

    out = two_stage(input, num_class, keep_prob, initial_channel)

    return out

if __name__ == "__main__":
    pass