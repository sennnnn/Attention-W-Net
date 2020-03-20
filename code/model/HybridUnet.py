
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

def spatial_attention_block(input):
    """
    spatial transformer attention block.
    Args:
        input:tensor that will be operated.
    Return:
        input:input * spatial weight.
    """
    c = input.get_shape().as_list()[-1]
    weight = CBR(input, c, kernel_size=1)
    weight = tf.nn.relu(weight)
    weight = CBR(input, 1, kernel_size=1)
    weight = tf.nn.sigmoid(weight)
    input = input*weight
    input = layers.batch_normalization(input, momentum=DECAY_BATCH_NORM, epsilon=EPSILON)
    
    return input

def APA(input):
    """
    artous convolution style pyramid attention block.
    Args:
        input:the input block.
    Return:
        output:the output block.
    """
    c = input.get_shape().as_list()[-1]
    raw = CBR(input, c, kernel_size=1)
    input = ACBR(input, c, 1)
    down1 = CBR(input, c, kernel_size=1)
    input = ACBR(input, c, 3)
    down2 = CBR(input, c, kernel_size=1)
    input = ACBR(input, c, 5)
    down3 = CBR(input, c, kernel_size=1)

    output = CBR(down3+down2, c, kernel_size=1)
    output = CBR(output+down1, c, kernel_size=1)
    output = CBR(output+raw, c, kernel_size=1)

    return output

def hybridAttentionBlock(input):
    sub1 = channel_attention_block(input)
    sub1 = spatial_attention_block(sub1)
    sub2 = channel_attention_block(input)
    sub3 = spatial_attention_block(input)

    out = sub1+sub2+sub3+input
    out = layers.batch_normalization(out, momentum=DECAY_BATCH_NORM, epsilon=EPSILON)

    return out

def HybridAttentionUnet(input, num_class, keep_prob=0.5, initial_channel=64):
    """
    absorb many merits
    Args:
        input: the network input.
        num_class: the last layer output channel.
        keep_prob: the dropout layer keep probability.
        initial_channel: the network channel benchmark.
    Return:
        out: the network output.
    """
    c = initial_channel
    input = CBR(input, c)
    input = hybridAttentionBlock(input)
    input = CBR(input, c)
    fuse1 = input
    input = CBR(input, c, 2)

    c = c*2
    input = CBR(input, c)
    input = hybridAttentionBlock(input)
    input = CBR(input, c)
    fuse2 = input
    input = CBR(input, c, 2)

    c = c*2
    input = CBR(input, c)
    input = hybridAttentionBlock(input)
    input = CBR(input, c)
    fuse3 = input
    input = CBR(input, c, 2)

    c = c*2
    input = CBR(input, c)
    input = hybridAttentionBlock(input)
    input = CBR(input, c)
    fuse4 = input
    input = CBR(input, c, 2)

    c = c*2
    input = CBR(input, c)
    input = tf.nn.dropout(input, keep_prob=keep_prob) 
    input = APA(input)
    input = upsampling(input, c)

    c = c//2
    input = tf.concat([fuse4, input], axis=-1)
    input = CBR(input, c)
    input = hybridAttentionBlock(input)
    input = CBR(input, c)
    input = upsampling(input, c)

    c = c//2
    input = tf.concat([fuse3, input], axis=-1)
    input = CBR(input, c)
    input = hybridAttentionBlock(input)
    input = CBR(input, c)
    input = upsampling(input, c)

    c = c//2
    input = tf.concat([fuse2, input], axis=-1)
    input = CBR(input, c)
    input = hybridAttentionBlock(input)
    input = CBR(input, c)
    input = upsampling(input, c)

    c = c//2
    input = tf.concat([fuse1, input], axis=-1)
    input = CBR(input, c)
    input = hybridAttentionBlock(input)
    input = CBR(input, c)

    input = CBR(input, num_class, kernel_size=1)

    return input

if __name__ == '__main__':
    input = tf.placeholder(tf.float32, [None, 224, 224, 32])
    print(channel_attention_block(input))
    # print(ACBR(input, 3, 1))
    # print(hybridAttentionUnet(input, 6))