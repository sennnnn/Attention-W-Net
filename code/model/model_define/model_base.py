import tensorflow as tf
import tensorflow.layers as layers

# layer parameters
DECAY_BATCH_NORM = 0.9
EPSILON = 1E-05
LEAKY_RELU = 0.1

def C(input, filters, strides=1, kernel_size=3):
    """
    convolution only
    """
    input = layers.conv2d(input, filters, kernel_size, use_bias=True ,strides=strides, padding='same', \
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))

    return input

def CB(input, filters, strides=1, kernel_size=3):
    """
    convolution + batch normalization
    """
    input = layers.conv2d(input, filters, kernel_size, use_bias=True ,strides=strides, padding='same', \
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
    input = layers.batch_normalization(input, momentum=DECAY_BATCH_NORM, epsilon=EPSILON)

    return input

def CBR(input, filters, strides=1, kernel_size=3):
    """
    convolution + batch normalization + leaky relu operation
    """
    input = layers.conv2d(input, filters, kernel_size, use_bias=True ,strides=strides, padding='same', \
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
    input = layers.batch_normalization(input, momentum=DECAY_BATCH_NORM, epsilon=EPSILON)
    input = tf.nn.leaky_relu(input, alpha=LEAKY_RELU, name='ac')
    
    return input

def AC(input, filters, rate, kernel_size=3):
    """
    atrous convolution
    """
    c = input.get_shape().as_list()[-1]
    filters_variable = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, c, filters], dtype=tf.float32))
    input = tf.nn.atrous_conv2d(input, filters_variable, rate, padding='SAME')

    return input

def ACB(input, filters, rate, kernel_size=3):
    """
    atrous convolution + batch normalization
    """
    input = AC(input, filters, rate, kernel_size)
    input = layers.batch_normalization(input, momentum=DECAY_BATCH_NORM, epsilon=EPSILON)

    return input

def ACBR(input, filters, rate, kernel_size=3):
    """
    atrous convolution + batch normalization
    """
    input = AC(input, filters, rate, kernel_size)
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
    """
    # Up-sampling Layer,implemented by transpose convolution.
    input = layers.conv2d_transpose(input, filters, kernel_size, strides, padding='same')
    input = layers.batch_normalization(input, momentum=DECAY_BATCH_NORM, epsilon=EPSILON)

    return input

def construct_network(net, inputs, num_class, initial_channel=64, keep_prob=0.1):
    """
    constructing all kinds of unet-class network.
    net function args must be like this (input, num_class, initial_channel, keep_prob).
    """
    with tf.variable_scope('network'):
        predict = net(inputs, num_class=num_class, keep_prob=keep_prob, initial_channel=initial_channel)
    predict = tf.identity(predict, name='predict')
    softmax = tf.nn.softmax(predict, name='predict_softmax')
    argmax = tf.argmax(softmax, axis=-1, name='predict_argmax')
