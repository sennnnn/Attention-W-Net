import tensorflow as tf

def load_graph(frozen_graph_filename):

    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,name="")

    return graph

def load_graph(frozen_graph_filename):

    with open(frozen_graph_filename,"rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # 如果不自己设置默认计算图的话，那么import_graph_def将导入到运行脚本初始建立的哪个计算图
    with tf.Graph().as_default() as graph:
        # name是前缀，意思就是所有从graph_def中导入的计算图ops和variables都将是带有(name)/前缀的
        tf.import_graph_def(graph_def,name="")

    return graph

if __name__ == "__main__":
    graph_gfile_ops = load_graph("frozen_model/gfile_save_ops.pb")
    graph_gfile_all = load_graph("frozen_model/gfile_save_all.pb")
    graph_system_ops = load_graph("frozen_model/system_save_ops.pb")
    graph_system_all = load_graph("frozen_model/system_save_all.pb")
    # 变量到常量会损失至少两个ops，变量一般有:initial_value、assign、变量本身、read四个op，而常量只有:常量本身、read两个op
    print(len(graph_gfile_ops.as_graph_def().node),len(graph_gfile_all.as_graph_def().node))
    
    with tf.Session(graph=graph_gfile_ops) as sess:
        default_graph = tf.get_default_graph() 
        ops = default_graph.get_operations()
        [print(x) for x in ops]
        # print(len(ops))
        # print(len(tf.global_variables()))
        # get_tensor_by_name实际上就是获得ops的输出而已，而tensor名字一般就是<op_name>:<output_index>
        # :0是因为这个ops只有一个输出，若有两个输出那么tensor<"b:0">与tensor<"b:1">都存在，且二者
        # 一个变量下面有四个ops，这里b还是变量，所以要经历
        # b/initial_value->b/assign而后才能是b可读
        print(default_graph.get_tensor_by_name('b/read:0'))
        print(default_graph.get_tensor_by_name("b:0"))
        a,b = default_graph.get_tensor_by_name('b/initial_value:0'),default_graph.get_tensor_by_name("b/Assign:0")
        print(sess.run([a,b]))

    with tf.Session(graph=graph_gfile_all) as sess:
        default_graph = tf.get_default_graph() 
        ops = default_graph.get_operations()
        [print(x) for x in ops]
        # 此时b已经是常量了，所以就没有b/initial_value与b/assign这个ops了
        print(default_graph.get_tensor_by_name('b/read:0'))
        print(default_graph.get_tensor_by_name("b:0"))
        a,b = default_graph.get_tensor_by_name('b:0'),default_graph.get_tensor_by_name("b/read:0")
        print(sess.run([a,b]))

    # 下面的例子证明，其实只用普通的open也能对权重文件进行存取，下两种情况和上两种情况对应，且完全相同
    with tf.Session(graph=graph_system_ops) as sess:
        default_graph = tf.get_default_graph() 
        ops = default_graph.get_operations()
        [print(x) for x in ops]
        print(default_graph.get_tensor_by_name('b/read:0'))
        print(default_graph.get_tensor_by_name("b:0"))
        a,b = default_graph.get_tensor_by_name('b/initial_value:0'),default_graph.get_tensor_by_name("b/Assign:0")
        print(sess.run([a,b]))

    with tf.Session(graph=graph_system_all) as sess:
        default_graph = tf.get_default_graph() 
        ops = default_graph.get_operations()
        [print(x) for x in ops]
        print(default_graph.get_tensor_by_name('b/read:0'))
        print(default_graph.get_tensor_by_name("b:0"))
        a,b = default_graph.get_tensor_by_name('b:0'),default_graph.get_tensor_by_name("b/read:0")
        print(sess.run([a,b]))
