import tensorflow as tf
import tensorflow.layers as layers

# layer parameters
DECAY_BATCH_NORM = 0.9
EPSILON = 1E-05
LEAKY_RELU = 0.1

def AC(input, filters, rate):
    """
    atrous convolution
    Args:
        input:tensor that will be operated.
        filters:convolutional kernel channel size.
        rate:the convolution kernel expandation rate.
    Return:
        input:tensor that has been operated.
    """
    c = input.get_shape().as_list()[-1]
    filters_variable = tf.Variable(tf.truncated_normal([3, 3, c, filters], dtype=tf.float32))
    input = tf.nn.atrous_conv2d(input, filters_variable, rate, padding='SAME')

    return input

def C(input, filters, strides=1, kernel_size=3):
    """
    convolution only
    Args:
        input:tensor that will be operated.
        filters:convolutional kernel channel size.
        strides:the convolutional kernel move length of one calculation.
        kernel_size:convolutional kernel size.
    Return:
        input:tensor that has been operated.
    """
    input = layers.conv2d(input, filters, kernel_size, use_bias=True ,strides=strides, padding='same', \
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))

    return input

def ACB(input, filters, rate):
    """
    atrous convolution + batch normalization
    Args:
        input:tensor that will be operated.
        filters:convolutional kernel channel size.
        rate:the convolution kernel expandation rate.
    Return:
        input:tensor that has been operated.
    """
    input = AC(input, filters, rate)
    input = layers.batch_normalization(input, momentum=DECAY_BATCH_NORM, epsilon=EPSILON)

    return input

def CB(input, filters, strides=1, kernel_size=3):
    """
    convolution + batch normalization
    Args:
        input:tensor that will be operated.
        filters:convolutional kernel channel size.
        strides:the convolutional kernel move length of one calculation.
        kernel_size:convolutional kernel size.
    Return:
        input:tensor tha has been operated.
    """
    input = layers.conv2d(input, filters, kernel_size, use_bias=True ,strides=strides, padding='same', \
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
    input = layers.batch_normalization(input, momentum=DECAY_BATCH_NORM, epsilon=EPSILON)

    return input

def CBR(input, filters, strides=1, kernel_size=3):
    """
    convolution + batch normalization + leaky relu operation
    Args:
        input:tensor that will be operated.
        filters:convolutional kernel channel size.
        strides:the convolutional kernel move length of one calculation.
        kernel_size:convolutional kernel size.
    Return:
        input:tensor tha has been operated.
    """
    input = layers.conv2d(input, filters, kernel_size, use_bias=True ,strides=strides, padding='same', \
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
    input = layers.batch_normalization(input, momentum=DECAY_BATCH_NORM, epsilon=EPSILON)
    input = tf.nn.leaky_relu(input, alpha=LEAKY_RELU, name='ac')
    
    return input

def ACBR(input, filters, rate):
    """
    atrous convolution + batch normalization
    Args:
        input:tensor that will be operated.
        filters:convolutional kernel channel size.
        rate:the convolution kernel expandation rate.
    Return:
        input:tensor that has been operated.
    """
    input = AC(input, filters, rate)
    input = layers.batch_normalization(input, momentum=DECAY_BATCH_NORM, epsilon=EPSILON)
    input = tf.nn.leaky_relu(input, alpha=LEAKY_RELU, name='ac')
    
    return input

def res_block(input, filters):
    # General residual block
    shortcut = input
    input = CBR(input, filters)
    input = CBR(input, filters)

    return input + shortcut

def artous_conv(input, filters, rate):
    # Artous/Dilated Convolution，Only defined in tf,nn module.
    origin_channels = input.get_shape().as_list()
    weights = tf.Variable(tf.random_normal(shape=[3, 3, origin_channels, filters]))
    input = tf.nn.atrous_conv2d(input, weights, rate, "valid")

    return input

def bottle_neck_res_block(input, filters):
    # It can let your network deeper with less parameters.
    shortcut = input
    input = CBR(input, filters//4, 1)
    input = CBR(input, filters//4, 3)
    input = CBR(input, filters, 1)

    return input + shortcut

def upsampling(input, filters, kernel_size=3, strides=2):
    """
    convolution_transpose + batch normalization
    Args:
        input:the tensor that will be operated.
        filters:the convolutional kernel channel length.
        kernel_size:the transpose convolution kernel size.
        strides:moving length of one calculation.
    Return:
        input:the tensor that has been operated.
    """
    # Up-sampling Layer,implemented by transpose convolution.
    input = layers.conv2d_transpose(input, filters, kernel_size, strides, padding='same')
    input = layers.batch_normalization(input, momentum=DECAY_BATCH_NORM, epsilon=EPSILON)

    return input

def output_layer(input,thresh):
    """
    Args:
    input:the softmax output of the net.
    thresh:the threshould to binarilize the input.

    Return:
    out:Binary predict.
    """
    out = tf.cast((input > thresh), tf.uint8)

    return out

def construct_network(net, inputs, num_class, initial_channel=64, keep_prob=0.1):
    """
    Constructing all kinds of unet network.
    Args:
        net:network implementation function.
        num_class:the channel length of the output tensor.
        initial_channel:the first convolutional operation channel.
        inputs:network implementation function input args.
    Return:
        None
    """
    with tf.variable_scope('network'):
        predict = net(inputs, num_class=num_class, keep_prob=keep_prob, initial_channel=initial_channel)
    predict = tf.identity(predict, name='predict')
    softmax = tf.nn.softmax(predict, name='predict_softmax')
    argmax = tf.argmax(softmax, axis=-1, name='predict_argmax')

if __name__ == "__main__":
    """
    yeah~
    """
    num_class = 6
    input_1 = tf.placeholder(tf.float32,[None, None, None, 1])
    input_2 = tf.placeholder(tf.float32,[None, 224, 384, 1])
    input_3 = tf.placeholder(tf.float32,[None, 224, 384, 1])
    print(channel_attention_block(input_1))
    # input = r2Unet(input_1, 6)