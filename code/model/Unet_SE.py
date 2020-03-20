
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

def encoder_unet_SE(input, initial_channel=64, encoder_time=4):
        # the encoder part of the unet architecture.
## Encoder ##
    bc = initial_channel
    ret_list = []
    for i in list(range(encoder_time)):
        channel_base = initial_channel*2**i
        input = CBR(input, channel_base)
        input = channel_attention_block(input)
        input = CBR(input, channel_base)
        ret_list.append(input)
        input = CBR(input, channel_base, 2)

    ret_list.append(input)
## ##

    return tuple(ret_list)

def middle_deliver_layer_unet_SE(input, keep_prob, initial_channel=64, encoder_time=4):
    input = CBR(input, initial_channel*2**encoder_time)
    # To avoid over-fitting.
    input = channel_attention_block(input)
    input = tf.nn.dropout(input, keep_prob)
    input = CBR(input, initial_channel*2**encoder_time)

    return input

def decoder_unet_SE(input, fuse_list, initial_channel=64, decoder_time=4):
    for i in list(range(decoder_time))[::-1]:
        channel_base = initial_channel*2**i
        input = upsampling(input, channel_base*2)
        input = tf.concat([fuse_list[i], input], axis=-1)
        input = CBR(input, channel_base)
        input = channel_attention_block(input)
        input = CBR(input, channel_base)
    
    return input

def unet_output_layer(input, num_class):
    """
    The last layer of the unet-relating network.
    It can let output channel to be epual with the number of classes.
    """
    out = CBR(input, num_class, kernel_size=3)

    return out

def Unet_SE(input, num_class, keep_prob=0.1, initial_channel=64, ifout=True, encoder_decoder_time=4):
    # Attention mechanism block will be useful to face multiple segementation object.
## encoder ##
    fuse_list = encoder_unet_SE(input, initial_channel, encoder_decoder_time)
## ##

    input = middle_deliver_layer_unet_SE(fuse_list[-1], keep_prob, initial_channel, encoder_decoder_time)

## Decoder ##
    input = decoder_unet_SE(input, fuse_list[:encoder_decoder_time], initial_channel, encoder_decoder_time)
## ##
    
## output layer ##
    if(ifout):
        input = unet_output_layer(input, num_class)
## ##

    return input

## output layer ##
    if(ifout):
        input = unet_output_layer(input, num_class)
## ##

    return input