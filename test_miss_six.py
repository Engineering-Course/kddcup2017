import tensorflow as tf
import numpy as np
import os
import time
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Parameters
BATCH_SIZE = 1
SNAPSHOT_DIR = './checkpoint/useful_six'
DATA_DIR = './dataset/dataSets/testing_six'
TEST_ONE = False

# Network Parameters
n_hidden_1 = 9 # 1st layer number of features
n_hidden_2 = 9 # 2nd layer number of features
n_hidden_3 = 18
n_input = 9
n_classes = 1


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=0.1)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=0.1)),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], stddev=0.1)),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes], stddev=0.1))
}
biases = {
    'b1': tf.Variable(tf.constant(0.0, shape=[n_hidden_1])),
    'b2': tf.Variable(tf.constant(0.0, shape=[n_hidden_2])),
    'b3': tf.Variable(tf.constant(0.0, shape=[n_hidden_3])),
    'out': tf.Variable(tf.constant(0.0, shape=[n_classes]))
}

def network(data, name):
  with tf.variable_scope(name) as scope:

    layer_1 = tf.add(tf.matmul(data, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer


def main(sess):
    data = tf.placeholder(tf.float32, [BATCH_SIZE, n_input], name='data')
    pred = network(data, 'nn')

    tf.global_variables_initializer().run()

    # Saver for storing checkpoints of the model.
    # Restore variables
    restore_var = tf.global_variables()
    loader = tf.train.Saver(var_list=restore_var)
    if load(loader, sess, SNAPSHOT_DIR):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")    


    ##  test for one data
    if TEST_ONE:
        #  input data 
        link = 165.0
        # t1 = 102.83
        # t2 = 106.34
        # t3 = 141.43
        # t4 = 73.72
        # t5 = 80.27
        # t6 = 130.12
        hour = 115.0
        day = 280
        x_ = np.array([260.66,142.09237671,112.49593353,122.97939301,120.63757324,119.17584229,119.26552582,115.0,280])
        res = sess.run(pred, feed_dict={data: [x_]})
        print res

    ##  test for pipeline
    else:
        for root, dirs, files in os.walk(DATA_DIR):
            for fname in files:
                test_file = os.path.join(root, fname)
                x_ = []
                y_ = []            
                with open(test_file, 'r') as fr:
                    line_ = fr.readline()
                data_ = line_.split(',')
                item_x = [float(tt) for tt in data_]
                x_.append(item_x)

                for win in xrange(6):
                    res = sess.run(pred, feed_dict={data: x_})
                    y_.append(round(res[0][0], 2))
                    for jj in xrange(1, 6):
                        x_[0][jj] = x_[0][jj+1]
                    x_[0][6] = res[0][0]
                fi = open('result_six/{}'.format(fname), 'w')
                for ii in xrange(6):
                    fi.write(str(y_[ii]) + ' ')
                fi.close()



if __name__ == '__main__':

    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        main(sess)




