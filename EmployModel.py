# coding:utf8
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# 导入数据，创建一个session对象　，之后的运算都会跑在这个session里
# mnist = input_data.read_data_sets("TestEmploy/", one_hot=True)
sess = tf.Session()

# 加载模型
saver = tf.train.import_meta_graph('./save/-19.meta')
saver.restore(sess, tf.train.latest_checkpoint('./save'))

graph = tf.get_default_graph()
name_list = [n.name for n in tf.get_default_graph().as_graph_def().node]
print(name_list)

x_ = graph.get_tensor_by_name("x:0")
keep_prob_ = graph.get_tensor_by_name("keep_prob:0")

# Now, access the op that you want to run.
# y_ = graph.get_tensor_by_name("y_conv:0")
Att_v_max_ = graph.get_tensor_by_name('partial_softmax/Att_v_max')
Prediction_Softmax = graph.get_tensor_by_name('Softmax')

for i in range(20):
    batch = mnist.train.next_batch(1)
    # 预测
    y_conv = sess.run([Att_v_max_,Prediction_Softmax], feed_dict={x_: batch[0], keep_prob_: 0.5})  #
    correct_predition = tf.equal(tf.argmax(y_conv, 1), tf.argmax(batch[1], 1))
    correct_predition.eval(session=sess)
