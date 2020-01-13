import os
import time
import datetime
import numpy as np
import tensorflow as tf
import SimpleITK as stk

def readNiiAll(nii_path):
    image = stk.ReadImage(nii_path)
    array = stk.GetArrayFromImage(image)
    return image.GetSpacing(),image.GetOrigin(),array

def readImage(nii_path):
    image = stk.ReadImage(nii_path)
    return stk.GetArrayFromImage(image)

def saveAsNiiGz(numpy_array,nii_path,spacing,origin):
    image = stk.GetImageFromArray(numpy_array)
    image.SetSpacing(spacing);image.SetOrigin(origin)
    stk.WriteImage(image,nii_path)
    print("nii file is saved as {}".format(nii_path))

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

    print("load graph {} ...".format(frozen_graph_filename))
    return graph

def restore_from_pb(sess,frozen_graph,meta_graph):
    # frozen_graph 与 meta_graph 应该是相互匹配的
    ops = frozen_graph.get_operations()
    ops_restore = [x.name.replace('/read','') for x in ops if('/read' in x.name)]
    tensors_constant = [frozen_graph.get_tensor_by_name(x+':0') for x in ops_restore]
    tensors_variables = [meta_graph.get_tensor_by_name(x+':0') for x in ops_restore]
    do_list = []
    # [print(x.name) for x in tensors_constant]
    sess_local = tf.Session(graph=frozen_graph)
    for i in range(len(ops_restore)):
        temp = sess_local.run(tensors_constant[i])
        # print(i,' ',tensors_constant[i].name,'==>>',tensors_variables[i].name)
        # print
        # print(sess.run(tf.assign(tensors_variables[i],temp)))
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

def np_dice_index(a,b,delta=0.0001):
    a_area = np.sum(a[...,1:],dtype=np.float32)
    b_area = np.sum(b[...,1:],dtype=np.float32)
    cross = np.sum(a[...,1:]*b[...,1:],dtype=np.float32)
    return 2*cross/(a_area+b_area+delta)

def tf_dice(a,b,smooth=0.00001):
    a_area = tf.reduce_sum(a[...,1:])
    b_area = tf.reduce_sum(b[...,1:])
    a_area = tf.cast(a_area, tf.float32)
    b_area = tf.cast(b_area, tf.float32)
    cross_area = tf.reduce_sum(a[...,1:]*b[...,1:])
    cross_area = tf.cast(cross_area, tf.float32)
    return 2.*cross_area/(a_area+b_area+smooth)

def tf_dice_index_norm(a,b,num_class,delta=0.0001):
    # 输入为softmax则可以帮助one_hot编码之后再比较
    a = tf.argmax(a,axis=-1)
    a = tf.one_hot(a, num_class, 1, 0)
    b = tf.argmax(b,axis=-1)
    b = tf.one_hot(b, num_class, 1, 0)
    a_area = tf.reduce_sum(a[...,1:])
    b_area = tf.reduce_sum(b[...,1:])
    a_area = tf.cast(a_area, tf.float32)
    b_area = tf.cast(b_area, tf.float32)
    cross = tf.reduce_sum(a[...,1:]*b[...,1:])
    b_area = tf.cast(cross, tf.float32)
    return 2.*cross/(a_area+b_area+delta)


def grayToRgb(mask):
    # 将掩膜的灰度图像转换成RGB图像
    w,h = mask.shape
    rgb_mask = np.zeros((w,h,3),dtype=np.uint8)
    keys = list(names.keys())
    for i in range(w):
        for j in range(h):
            rgb_mask[i,j] = stringToHex(names[keys[mask[i,j]]])
    return rgb_mask    

def stringToHex(string):
    meta = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'A':10,'B':11,'C':12,'D':13,'E':14,'F':15}
    string = string[1:]
    r = meta[string[0]]*16+meta[string[1]]
    g = meta[string[2]]*16+meta[string[3]]
    b = meta[string[4]]*16+meta[string[5]]
    return (r,g,b)

names = {
'black' :               '#000000',

'aliceblue':            '#F0F8FF',

'orange':               '#FFA500',

'darkviolet':           '#9400D3',

'mediumvioletred':      '#C71585',

'aliceblue':            '#F0F8FF',

'antiquewhite':         '#FAEBD7',

'aqua':                 '#00FFFF',

'aquamarine':           '#7FFFD4',

'azure':                '#F0FFFF',

'beige':                '#F5F5DC',

'bisque':               '#FFE4C4',

'black':                '#000000',

'blanchedalmond':       '#FFEBCD',

'blue':                 '#0000FF',

'blueviolet':           '#8A2BE2',

'brown':                '#A52A2A',

'burlywood':            '#DEB887',

'cadetblue':            '#5F9EA0',

'chartreuse':           '#7FFF00',

'chocolate':            '#D2691E',

'coral':                '#FF7F50',

'cornflowerblue':       '#6495ED',

'cornsilk':             '#FFF8DC',

'crimson':              '#DC143C',

'cyan':                 '#00FFFF',

'darkblue':             '#00008B',

'darkcyan':             '#008B8B',

'darkgoldenrod':        '#B8860B',

'darkgray':             '#A9A9A9',

'darkgreen':            '#006400',

'darkkhaki':            '#BDB76B',

'darkmagenta':          '#8B008B',

'darkolivegreen':       '#556B2F',

'darkorange':           '#FF8C00',

'darkorchid':           '#9932CC',

'darkred':              '#8B0000',

'darksalmon':           '#E9967A',

'darkseagreen':         '#8FBC8F',

'darkslateblue':        '#483D8B',

'darkslategray':        '#2F4F4F',

'darkturquoise':        '#00CED1',

'darkviolet':           '#9400D3',

'deeppink':             '#FF1493',

'deepskyblue':          '#00BFFF',

'dimgray':              '#696969',

'dodgerblue':           '#1E90FF',

'firebrick':            '#B22222',

'floralwhite':          '#FFFAF0',

'forestgreen':          '#228B22',

'fuchsia':              '#FF00FF',

'gainsboro':            '#DCDCDC',

'ghostwhite':           '#F8F8FF',

'gold':                 '#FFD700',

'goldenrod':            '#DAA520',

'gray':                 '#808080',

'green':                '#008000',

'greenyellow':          '#ADFF2F',

'honeydew':             '#F0FFF0',

'hotpink':              '#FF69B4',

'indianred':            '#CD5C5C',

'indigo':               '#4B0082',

'ivory':                '#FFFFF0',

'khaki':                '#F0E68C',

'lavender':             '#E6E6FA',

'lavenderblush':        '#FFF0F5',

'lawngreen':            '#7CFC00',

'lemonchiffon':         '#FFFACD',

'lightblue':            '#ADD8E6',

'lightcoral':           '#F08080',

'lightcyan':            '#E0FFFF',

'lightgoldenrodyellow': '#FAFAD2',

'lightgreen':           '#90EE90',

'lightgray':            '#D3D3D3',

'lightpink':            '#FFB6C1',

'lightsalmon':          '#FFA07A',

'lightseagreen':        '#20B2AA',

'lightskyblue':         '#87CEFA',

'lightslategray':       '#778899',

'lightsteelblue':       '#B0C4DE',

'lightyellow':          '#FFFFE0',

'lime':                 '#00FF00',

'limegreen':            '#32CD32',

'linen':                '#FAF0E6',

'magenta':              '#FF00FF',

'maroon':               '#800000',

'mediumaquamarine':     '#66CDAA',

'mediumblue':           '#0000CD',

'mediumorchid':         '#BA55D3',

'mediumpurple':         '#9370DB',

'mediumseagreen':       '#3CB371',

'mediumslateblue':      '#7B68EE',

'mediumspringgreen':    '#00FA9A',

'mediumturquoise':      '#48D1CC',

'mediumvioletred':      '#C71585',

'midnightblue':         '#191970',

'mintcream':            '#F5FFFA',

'mistyrose':            '#FFE4E1',

'moccasin':             '#FFE4B5',

'navajowhite':          '#FFDEAD',

'navy':                 '#000080',

'oldlace':              '#FDF5E6',

'olive':                '#808000',

'olivedrab':            '#6B8E23',

'orange':               '#FFA500',

'orangered':            '#FF4500',

'orchid':               '#DA70D6',

'palegoldenrod':        '#EEE8AA',

'palegreen':            '#98FB98',

'paleturquoise':        '#AFEEEE',

'palevioletred':        '#DB7093',

'papayawhip':           '#FFEFD5',

'peachpuff':            '#FFDAB9',

'peru':                 '#CD853F',

'pink':                 '#FFC0CB',

'plum':                 '#DDA0DD',

'powderblue':           '#B0E0E6',

'purple':               '#800080',

'red':                  '#FF0000',

'rosybrown':            '#BC8F8F',

'royalblue':            '#4169E1',

'saddlebrown':          '#8B4513',

'salmon':               '#FA8072',

'sandybrown':           '#FAA460',

'seagreen':             '#2E8B57',

'seashell':             '#FFF5EE',

'sienna':               '#A0522D',

'silver':               '#C0C0C0',

'skyblue':              '#87CEEB',

'slateblue':            '#6A5ACD',

'slategray':            '#708090',

'snow':                 '#FFFAFA',

'springgreen':          '#00FF7F',

'steelblue':            '#4682B4',

'tan':                  '#D2B48C',

'teal':                 '#008080',

'thistle':              '#D8BFD8',

'tomato':               '#FF6347',

'turquoise':            '#40E0D0',

'violet':               '#EE82EE',

'wheat':                '#F5DEB3',

'white':                '#FFFFFF',

'whitesmoke':           '#F5F5F5',

'yellow':               '#FFFF00',

'yellowgreen':          '#9ACD32'}
