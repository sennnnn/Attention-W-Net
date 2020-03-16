
from .Unet import *

def unet_input_fuse(inputs, num_class=None, keep_prob=0.1, initial_channel=64):
    assert num_class != None, 'Error, num_class can not be None.'
    input_all = tf.concat([input for input in inputs], axis=-1)
    out = unet([input_all], num_class=num_class, keep_prob=keep_prob, initial_channel=initial_channel, ifout=False)
    out = unet_output_layer(out, num_class)

    return out

def unet_middle_fuse(inputs, num_class=None, keep_prob=0.1, initial_channel=64):
    assert num_class != None, 'Error, num_class can not be None.'
## Encoder ##    
    bc = initial_channel
    input_num = len(inputs)
    encoder_list = []
    # the purpose of [::-1] is convert (fuse1, fuse2, fuse3, fuse4, out) -> (out, fuse4, fuse3, fuse2, fuse1)
    for input in inputs:
        encoder_list.append(encoder_unet(input, bc))

## ##
    complex = tf.concat([meta[-1] for meta in encoder_list], axis=-1)
    complex = CBR(complex, bc*16)
    complex = tf.nn.dropout(complex, keep_prob)
    complex = CBR(complex, bc*16)

## Decoder ##
    iter_range = list(range(4))[::-1] 
    for i in iter_range:
        # 3, 2, 1, 0
        channel_base = bc*2**i
        complex = upsampling(complex, channel_base*2)
        fuse_list = [encoder[i] for encoder in encoder_list] + [complex]
        complex = tf.concat(fuse_list, axis=-1)
        complex = CBR(complex, channel_base)
        complex = CBR(complex, channel_base)
## ##

    complex = unet_output_layer(complex, num_class)

    return complex

def unet_output_fuse(inputs, num_class=None, keep_prob=0.1, initial_channel=64):
    assert num_class != None, 'Error, num_class can not be None.'
    out_list = []
    kwargs = {'num_class': num_class, 'keep_prob': keep_prob, 'initial_channel': initial_channel, 'ifout': False}
    for input in inputs:
        out = unet([input], **kwargs)
        out_list.append(out)

    out = tf.concat(out_list, axis=-1)
    out = unet_output_layer(out, num_class)
    
    return out
