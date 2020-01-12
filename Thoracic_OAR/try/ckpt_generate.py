import tensorflow as tf

def model_composing():
    x = tf.placeholder(tf.int32, name='x')
    y = tf.placeholder(tf.int32, name='y')
    b = tf.Variable(1, name='b')
    xy = tf.multiply(x, y)
    # 这里的输出需要加上name属性
    op = tf.add(xy, b, name='add')

# variable_scope 主要影响 tf.get_variable()的作用域，如果作用域不同，那么就算重名也是可以的，
with tf.variable_scope("function"):
    # model_composing()
    x = tf.placeholder(tf.int32, name='x')
    y = tf.placeholder(tf.int32, name='y')
    b = tf.Variable(1, name='b')
    xy = tf.multiply(x, y)
    # 这里的输出需要加上name属性
    op = tf.add(xy, b, name='add')
d = tf.get_variable(name='function/b', shape=[], dtype=tf.float32)
print(b,d)
# init = tf.global_variables_initializer()
# saver = tf.train.saver()
# with tf.Session() as sess:
#     sess.run(init)
#     default = tf.get_default_graph()
#     ops = default.get_operations()
#     [print(x) for x in ops]