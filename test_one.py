import tensorflow as tf
import numpy as np
import os
import time
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Parameters
BATCH_SIZE = 1
SNAPSHOT_DIR = './checkpoint/useful_one'
DATA_DIR = './result_combine/testing_six_stage5'
TEST_ONE = False

# Network Parameters
n_hidden_1 = 4 # 1st layer number of features
n_hidden_2 = 8 # 2nd layer number of features
n_input = 4
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

    if TEST_ONE:
        #  input data 
        link = 241.0
        t1 = 108.88
        hour = 250.0
        day = 160
        x_ = np.array([260.66,192.88,115.0,250])
        res = sess.run(pred, feed_dict={data: [x_]})
        print res
    else:
        for root, dirs, files in os.walk(DATA_DIR):
            for fname in files:
                test_file = os.path.join(root, fname)
                x_ = []
                y_ = []            
                with open(test_file, 'r') as fr:
                    line_ = fr.readline()
                data_ = line_.split(' ')
                data_ = data_[:-1]
                item_x = [float(tt) for tt in data_]
                x_.append(item_x)
                del x_[0][1:6]

                res = sess.run(pred, feed_dict={data: x_})

                with open('result_combine/stage_6/one/{}'.format(fname), 'w') as fi:
                    fi.write(str(res[0][0]) + ' ')


if __name__ == '__main__':

    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        main(sess)


