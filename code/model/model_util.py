import tensorflow as tf

def load_graph(graph_def_filename):
    """
    To get graph from graph_def file.
    Args:
        graph_def_filename:The graph_def file save path.
    Return:
        graph:The graph loaded from graph_def file.
    """
    with tf.gfile.GFile(graph_def_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    print("load graph {} ...".format(graph_def_filename))

    return graph

def restore_from_pb(sess, frozen_graph, meta_graph):
    """
    Ckpt restore from frozen model file.
    Args:
        sess:The Session that is connected with meta_graph.
        frozen_graph:The graph which will be used to restore.
        meta_graph:The graph which will be restored.
    Return:
        sess:The Session after processing.
    """ 
    # frozen_graph 与 meta_graph 应该是相互匹配的
    ops = frozen_graph.get_operations()
    ops_restore = [x.name.replace('/read','') for x in ops if('/read' in x.name)]
    tensors_constant = [frozen_graph.get_tensor_by_name(x+':0') for x in ops_restore]
    tensors_variables = [meta_graph.get_tensor_by_name(x+':0') for x in ops_restore]
    do_list = []
    sess_local = tf.Session(graph=frozen_graph)
    with tf.variable_scope('frozen_save'):
        for i in range(len(ops_restore)):
            try:
                temp = sess_local.run(tensors_constant[i])
                op = tf.assign(tensors_variables[i], temp)
                do_list.append(op)
            except:
                print('{} => {} error, frozen graph tensor name {} => meta graph tensor name {}'\
                      .format(tensors_constant[i].get_shape(), tensors_variables[i].get_shape(), 
                              tensors_constant[i].name, tensors_variables[i].name))
                exit()
        sess_local.close()
        sess.run(do_list)

    return sess

def frozen_graph(sess, output_graph_path):
    """
    Extracting the sub graph of the current computing graph.
    Args:
        sess:The current session.
        output_graph_path:The frozen graph file save path.
    Return:
        string:The saving information.
    """
    # 因为计算图上只有ops没有变量，所以要通过会话，来获取那些变量
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, 
                                                                   tf.get_default_graph().as_graph_def(),
                                                                   ["predict"])

    with open(output_graph_path, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    return "{} ops written to {}.\n".format(len(output_graph_def.node), output_graph_path)