import os
import time
import datetime
import numpy as np
import tensorflow as tf

def get_newest(dir_path):
    file_list = os.listdir(dir_path)
    newest_file = os.path.join(dir_path,file_list[0])
    for filename in file_list:
        one_file = os.path.join(dir_path,filename)
        if(get_ctime(newest_file) < get_ctime(one_file)):
            newest_file = one_file
    return newest_file 

def get_ctime(file_path,ifstamp=True):
    if(ifstamp):
        return os.path.getctime(file_path)
    else:
        timeStruct = time.localtime(os.path.getctime(file_path))
        return time.strftime("%Y-%m-%d %H:%M:%S",timeStruct)

def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    return graph

def restore_from_pb(sess,frozen_graph,meta_graph):
    # frozen_graph 与 meta_graph 应该是相互匹配的
    ops = frozen_graph.get_operations()
    ops_restore = [x.name.replace('/read','') for x in ops if('/read' in x.name)]
    tensors_constant = [frozen_graph.get_tensor_by_name(x+':0') for x in ops_restore]
    tensors_variables = [meta_graph.get_tensor_by_name(x+':0') for x in ops_restore]
    do_list = []
    sess_local = tf.Session(graph=frozen_graph)
    for i in range(len(ops_restore)):
        temp = sess_local.run(tensors_constant[i])
        do_list.append(tf.assign(tensors_variables[i],temp))
    sess_local.close()
    sess.run(do_list)
    return sess

def frozen_graph(sess, output_graph):
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, # 因为计算图上只有ops没有变量，所以要通过会话，来获得变量有哪些
                                                                   tf.get_default_graph().as_graph_def(),
                                                                   ["segementation_result","dice"])
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