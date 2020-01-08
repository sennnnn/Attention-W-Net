import tensorflow as tf
import os

x = tf.placeholder(tf.int32, name='x')
y = tf.placeholder(tf.int32, name='y')
b = tf.Variable(1, name='b')
xy = tf.multiply(x, y)
# 这里的输出需要加上name属性
op = tf.add(xy, b, name='out')
op = tf.cast(op,tf.float32)
op = tf.nn.softmax(op)
op = tf.add(op,1,'real_out')

pb_file_path = os.getcwd()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # 注意节点node与变量的区别，节点不会带有:0的命名，而变量会，
    # 例如add为tf.add这个ops，而add:0则为tf.add返回的tensor
    # graph = tf.graph_util.convert_variables_to_constants()
    # 这个函数之所以需要是因为，如果仅仅只是tf.get_default_graph()，
    # 那么只能获取ops，而pb文件能够保存所有的ops和constant，所以需要
    # 经过上面的转换将所有变量转化为常量，这样就可以完全保存模型了。
    # 当一个tensorflow脚本执行时，就算其中没有任何内容，也会生成graph
    # 例如如下
    # import tensorflow as tf
    # print(tf.get_default_graph())
    # 会输出一个图对象
    # 而自己定义图要设置为默认那么就得:
    # graph_mine = tf.Graph()
    # with graph.as_default() as graph:
    #   ...... ，这里定义才会在graph下                
    if(not tf.gfile.IsDirectory("frozen_model")):
        tf.gfile.MkDir('frozen_model')

    if(not os.path.exists("frozen_model")):
        os.mkdir("frozen_model")

    with tf.gfile.GFile('frozen_model/gfile_save_ops.pb', 'wb') as f_t:
        graph_def = tf.get_default_graph().as_graph_def()
        f_t.write(graph_def.SerializeToString())

    with open("frozen_model/system_save_ops.pb", 'wb') as f_s:
        graph_def = tf.get_default_graph().as_graph_def()
        f_s.write(graph_def.SerializeToString())

    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, # 因为计算图上只有ops没有变量，所以要通过会话，来获得变量有哪些
                                                                   tf.get_default_graph().as_graph_def(),
                                                                   ["real_out"])
    if(not tf.gfile.IsDirectory("frozen_model")):
        tf.gfile.MkDir('frozen_model')

    if(not os.path.exists("frozen_model")):
        os.mkdir("frozen_model")
    
    with tf.gfile.GFile('frozen_model/gfile_save_all.pb', 'wb') as f_t:
        graph_def = output_graph_def
        f_t.write(graph_def.SerializeToString())

    with open("frozen_model/system_save_all.pb", 'wb') as f_s:
        graph_def = output_graph_def
        f_s.write(graph_def.SerializeToString())