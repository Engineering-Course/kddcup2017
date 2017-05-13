import tensorflow as tf
import numpy as np
import os
import time
from load_data import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Parameters
BATCH_SIZE = 1
NUM_STEPS = 84
SNAPSHOT_DIR = './checkpoint'

# Network Parameters
n_hidden_1 = 12 # 1st layer number of features
n_hidden_2 = 24 # 2nd layer number of features
n_hidden_3 = 48
n_input = 12
n_classes = 1



# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=0.1)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=0.1)),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], stddev=0.1)),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes], stddev=0.1))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], stddev=0.1)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2], stddev=0.1)),
    'b3': tf.Variable(tf.random_normal([n_hidden_3], stddev=0.1)),
    'out': tf.Variable(tf.random_normal([n_classes], stddev=0.1))
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

def save(saver, sess, logdir, step):
    '''Save weights.   
    Args:
     saver: TensorFlow Saver object.
     sess: TensorFlow session.
     logdir: path to the snapshots directory.
     step: current training step.
    '''
    if not os.path.exists(logdir):
        os.makedirs(logdir)   
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
      
    if not os.path.exists(logdir):
      os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(ckpt_path, ckpt_name))
        print("Restored model parameters from {}".format(ckpt_name))
        return True
    else:
        return False  

def main(sess):
    data = tf.placeholder(tf.float32, [BATCH_SIZE, n_input], name='data')
    pred = network(data, 'nn')

    tf.global_variables_initializer().run()

    # Saver for storing checkpoints of the model.
    # Restore variables
    restore_var = tf.global_variables()
    saver = tf.train.Saver(var_list=restore_var, max_to_keep=100)
    loader = tf.train.Saver(var_list=restore_var)
    if load(loader, sess, SNAPSHOT_DIR):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")    

    for step in range(NUM_STEPS):
        data_id = step * BATCH_SIZE
        x_ = load_test_travel_data(data_id, BATCH_SIZE)
        y_ = []
        # print x_[0]
        for win in xrange(6):
            res = sess.run(pred, feed_dict={data: x_})
            y_.append(round(res[0][0], 2))
            for jj in xrange(1, 6):
                x_[0][jj] = x_[0][jj+1]
            x_[0][6] = res[0][0]
        fi = open('result/{}.txt'.format(step), 'w')
        for ii in xrange(6):
            fi.write(str(y_[ii]) + ' ')
        fi.close()
            # print x_
            # wait = raw_input()



if __name__ == '__main__':

    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        main(sess)




