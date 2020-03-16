
from .model_base import *

def encoder_unet(input, initial_channel=64, encoder_time=4):
        # the encoder part of the unet architecture.
## Encoder ##
    bc = initial_channel
    ret_list = []
    for i in list(range(encoder_time)):
        channel_base = initial_channel*2**i
        input = CBR(input, channel_base)
        input = CBR(input, channel_base)
        ret_list.append(input)
        input = CBR(input, channel_base, 2)

    ret_list.append(input)
## ##

    return tuple(ret_list)

def middle_deliver_layer_unet(input, keep_prob, initial_channel=64, encoder_time=4):
    input = CBR(input, initial_channel*2**encoder_time)
    # To avoid over-fitting.
    input = tf.nn.dropout(input, keep_prob)
    input = CBR(input, initial_channel*2**encoder_time)

    return input

def decoder_unet(input, fuse_list, initial_channel=64, decoder_time=4):
    for i in list(range(decoder_time))[::-1]:
        channel_base = initial_channel*2**i
        input = upsampling(input, channel_base*2)
        input = tf.concat([fuse_list[i], input], axis=-1)
        input = CBR(input, channel_base)
        input = CBR(input, channel_base)
    
    return input

def unet_output_layer(input, num_class):
    """
    The last layer of the unet-relating network.
    It can let output channel to be epual with the number of classes.
    """
    out = CBR(input, num_class, kernel_size=3)

    return out

def unet(inputs, num_class, keep_prob=0.1, initial_channel=64, ifout=True, encoder_decoder_time=4):
    # Baseline Unet constructure.
## Encoder ##
    bc = initial_channel
    fuse_list = encoder_unet(inputs[0], initial_channel, encoder_decoder_time)
## ##

    input = middle_deliver_layer_unet(fuse_list[-1], keep_prob, initial_channel, encoder_decoder_time)

## Decoder ##
    input = decoder_unet(input, fuse_list[:encoder_decoder_time], initial_channel, encoder_decoder_time)
## ##

## output layer ##
    if(ifout):
        input = unet_output_layer(input, num_class)
## ##

    return input