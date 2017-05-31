import tensorflow as tf
import numpy as np
import os
import time
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Parameters
BATCH_SIZE = 1
SNAPSHOT_DIR = './checkpoint/useful_three'

# Network Parameters
n_hidden_1 = 6 # 1st layer number of features
n_hidden_2 = 12 # 2nd layer number of features
n_input = 6
n_classes = 1


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=0.1)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=0.1)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=0.1))
}
biases = {
    'b1': tf.Variable(tf.constant(0.0, shape=[n_hidden_1])),
    'b2': tf.Variable(tf.constant(0.0, shape=[n_hidden_2])),
    'out': tf.Variable(tf.constant(0.0, shape=[n_classes]))
}

def network(data, name):
  with tf.variable_scope(name) as scope:

    layer_1 = tf.add(tf.matmul(data, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

def main(sess):
    data = tf.placeholder(tf.float32, [BATCH_SIZE, n_input], name='data')

    pred = network(data, 'nn')

    tf.global_variables_initializer().run()

    # Restore variables
    restore_var = tf.global_variables()
    loader = tf.train.Saver(var_list=restore_var)
    if load(loader, sess, SNAPSHOT_DIR):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")    

    #  input data 
    link = 241.0
    # t1 = 102.83
    # t2 = 106.34
    # t3 = 141.43
    hour = 115.0
    day = 190
    x_ = np.array([260.66,192.88,162.90307617,154.25831604,115.0,250])
    res = sess.run(pred, feed_dict={data: [x_]})
    print res


if __name__ == '__main__':

    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        main(sess)


