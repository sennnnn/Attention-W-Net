import numpy as np
import tensorflow as tf

def load_graph(frozen_graph_filename):
    
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    return graph

def frozen_graph(sess, output_graph):
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, # 因为计算图上只有ops没有变量，所以要通过会话，来获得变量有哪些
                                                                   tf.get_default_graph().as_graph_def(),
                                                                   ["segementation_result"])
    with open(output_graph,"wb") as f:
        f.write(output_graph_def.SerializeToString())

    return "{} ops written to {}.\n".format(len(output_graph_def.node), output_graph)

def ifsmaller(price_list,price):
    if(len(price_list) == 0):
        return 0
    else:
        if(price <= price_list[-1]):
            return 1
        else:
            return 0

def iflarger(price_list,price):
    if(len(price_list) == 0):
        return 0
    else:
        if(price >= price_list[-1]):
            return 1
        else:
            return 0

def one_hot(nparray, depth=0, on_value=1, off_value=0):
    if depth == 0:
        depth = np.max(nparray) + 1
    # 深度应该符合one_hot条件，其实keras有to_categorical(data,n_classes,dtype=float..)弄成one_hot
    assert np.max(nparray) < depth, "the max index of nparray: {} is larger than depth: {}".format(np.max(nparray), depth)
    shape = nparray.shape
    out = np.ones((*shape, depth),np.uint8) * off_value
    indices = []
    for i in range(nparray.ndim):
        tiles = [1] * nparray.ndim
        s = [1] * nparray.ndim
        s[i] = -1
        r = np.arange(shape[i]).reshape(s)
        if i > 0:
            tiles[i-1] = shape[i-1]
            r = np.tile(r, tiles)
        indices.append(r)
    indices.append(nparray)
    out[tuple(indices)] = on_value
    return out

def dice(A,B,smooth=0.00001):
    A_area = tf.reduce_sum(A[...,1:])
    B_area = tf.reduce_sum(B[...,1:])
    cross_area = tf.reduce_sum(A[...,1:]*B[...,1:])
    return 2*cross_area/(A_area+B_area+smooth)