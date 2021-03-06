import tensorflow as tf
import numpy as np
import os
import time
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Parameters
window = 1
phrase = 'one'
BATCH_SIZE = 5
LEARNING_RATE = 1e-5
TRAINING_FLAG = False#True
EPOCH = 5000
# SNAPSHOT_DIR = './checkpoint/ck_{}'.format(phrase)
SNAPSHOT_DIR = './checkpoint/useful_one'
LOG_DIR = './logs/{}'.format(phrase)
TRAIN_ID_FILE = 'dataset/dataSets/train_{}_id.txt'.format(window)

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
    label = tf.placeholder(tf.float32, [BATCH_SIZE, n_classes], name='label')

    pred = network(data, 'nn')

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.abs(pred - label))
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

    # loss summary
    loss_sum = tf.summary.scalar("loss", cost)
    summary_writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())

    tf.global_variables_initializer().run()

    # Restore variables
    restore_var = tf.global_variables()
    saver = tf.train.Saver(var_list=restore_var, max_to_keep=5000)
    loader = tf.train.Saver(var_list=restore_var)
    if load(loader, sess, SNAPSHOT_DIR):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")    

    if TRAINING_FLAG:
        # Iterate over training steps.
        counter = 1
        fi = open('record/{}.txt'.format(phrase), 'w')
        for ee in xrange(EPOCH):
            with open(TRAIN_ID_FILE, 'r') as list_file:
                data_list = list_file.readlines()
            np.random.shuffle(data_list)
            batch_idxs = len(data_list) // BATCH_SIZE

            for idx in xrange(0, batch_idxs):
                batch_files = data_list[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
                x_, y_ = load_train_travel_data(batch_files, window)
                summary, _ = sess.run([loss_sum, optimizer], feed_dict={data: x_, label: y_})
                summary_writer.add_summary(summary, counter)
                counter += 1
            save(saver, sess, SNAPSHOT_DIR, counter)
            ##  Test...
            error = []
            for idx in xrange(0, batch_idxs):
                batch_files = data_list[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
                x_, y_ = load_train_travel_data(batch_files, window)        
                y, res = sess.run([label, pred], feed_dict={data: x_, label: y_})
                for bb in xrange(BATCH_SIZE):
                    error.append(np.abs(y[bb][0]-res[bb][0])/y[bb][0])
            test_result = np.mean(error)

            y, res = sess.run([label, pred], feed_dict={data: x_, label: y_})
            print('test_result: {:f}, epoch {:f}, step {:d}, y = {:.3f}, res = {:.3f}, loss = {:.3f}'.format(test_result, ee, counter,  y[0][0], res[0][0], np.abs(y[0][0]-res[0][0])/y[0][0]))
            fi.write('test_result: {:f}, epoch {:f}, step {:d}\n'.format(test_result, ee, counter))
        fi.close()
    else:
        error = []
        with open(TRAIN_ID_FILE, 'r') as list_file:
            data_list = list_file.readlines()
        batch_idxs = len(data_list) // BATCH_SIZE

        for idx in xrange(0, batch_idxs):
            batch_files = data_list[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
            x_, y_ = load_train_travel_data(batch_files, window)        
            y, res = sess.run([label, pred], feed_dict={data: x_, label: y_})
            for bb in xrange(BATCH_SIZE):
                error.append(np.abs(y[bb][0]-res[bb][0])/y[bb][0])
        print np.mean(error)



if __name__ == '__main__':

    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        main(sess)




