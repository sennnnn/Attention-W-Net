import numpy as np
import tensorflow as tf

def np_dice_index(a, b, delta=0.0001):
    a_area = np.sum(a[...,1:], dtype=np.float32)
    b_area = np.sum(b[...,1:], dtype=np.float32)
    cross = np.sum(a[...,1:]*b[...,1:], dtype=np.float32)

    return 2*cross/(a_area+b_area+delta)

def np_dice_index_channel_wise(a, b):
    return_list = []
    channel_length = a.shape[-1]
    for i in range(1, channel_length):
        return_list.append(np_dice_index(a[...,i], b[...,i]))

    return return_list

def tf_dice(a, b, smooth=0.00001):
    """
    The dice metric function implemented by using tensorflow module.
    Args:
        a:The tensor waiting to calculate the dice metric.
        b:The tensor waiting to calculate the dice metric.
        smooth:The constant which is set to avoid zero division.
    Return:
        temp:The dice metric between tensor a and tensor b.
    """
    a_area = tf.reduce_sum(a[...,1:])
    b_area = tf.reduce_sum(b[...,1:])
    a_area = tf.cast(a_area, tf.float32)
    b_area = tf.cast(b_area, tf.float32)
    cross_area = tf.reduce_sum(a[...,1:]*b[...,1:])
    cross_area = tf.cast(cross_area, tf.float32)

    return 2.*cross_area/(a_area+b_area+smooth)

def tf_dice_index_norm(a, b, num_class, delta=0.0001):
    # 输入为softmax则可以帮助one_hot编码之后再比较
    a = tf.argmax(a, axis=-1)
    a = tf.one_hot(a, num_class, 1, 0)
    b = tf.argmax(b, axis=-1)
    b = tf.one_hot(b, num_class, 1, 0)
    a_area = tf.reduce_sum(a[...,1:])
    b_area = tf.reduce_sum(b[...,1:])
    a_area = tf.cast(a_area, tf.float32)
    b_area = tf.cast(b_area, tf.float32)
    cross = tf.reduce_sum(a[...,1:]*b[...,1:])
    b_area = tf.cast(cross, tf.float32)

    return 2.*cross/(a_area+b_area+delta)

def channel_weighted_cross_entropy_loss(label, predict, weight):
    predict = tf.nn.softmax(predict)
    cross = -tf.reduce_sum(weight*label*tf.log(predict+3e-9),axis=-1)

    return cross

def IOU(label, predict, smooth=0.0001):
    p = predict[..., 1:]
    l = label[..., 1:]
    cross = p*l

    p_area = np.sum(p)
    l_area = np.sum(l)
    cross_area = np.sum(cross)

    return (cross_area + smooth)/(p_area + l_area - cross_area + smooth)

def DSC(label, predict, smooth=0.0001):
    p = predict[..., 1:]
    l = label[..., 1:]
    cross = p*l

    p_area = np.sum(p)
    l_area = np.sum(l)
    cross_area = np.sum(cross)

    return (2*cross_area + smooth)/(p_area + l_area + smooth)

def Sensity(label, predict, smooth=0.0001):
    p = predict[..., 1:]
    l = label[..., 1:]
    cross = p*l

    all_area = np.sum(np.ones_like(l, dtype=np.uint8))
    p_area = np.sum(p)
    l_area = np.sum(l)
    cross_area = np.sum(cross)

    TP = cross_area
    FN = all_area - p_area - l_area + cross_area

    return (TP + smooth)/(TP + FN + smooth) 

def Precision(label, predict, smooth=0.0001):
    p = predict[..., 1:]
    l = label[..., 1:]
    cross = p*l

    p_area = np.sum(p)
    cross_area = np.sum(cross)

    TP = cross_area
    FP = p_area - cross_area

    return (TP + smooth)/(TP + FP + smooth) if(p_area != 0) else 0

def F1_score(label, predict, smooth=0.0001):
    recall = Sensity(label, predict, smooth)
    precision = Precision(label, predict, smooth)

    return 2*(precision * recall + smooth)/(precision + recall + smooth)