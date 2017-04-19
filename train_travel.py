import tensorflow as tf
import numpy as np
import os
import time
from load_data import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Parameters
BATCH_SIZE = 10
NUM_STEPS = 13260 * 500
SAVE_PRED_EVERY = 13260 * 5
TOTAL_DATA = 13260
LEARNING_RATE = 1e-4
SNAPSHOT_DIR = './checkpoint'
LOG_DIR = './logs'

# Network Parameters
n_hidden_1 = 12 # 1st layer number of features
n_hidden_2 = 12 # 2nd layer number of features
n_hidden_3 = 16
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
    label = tf.placeholder(tf.float32, [BATCH_SIZE, n_classes], name='label')

    pred = network(data, 'nn')

    # Define loss and optimizer
    # tf.transpose(pred)
    cost = tf.reduce_mean(tf.abs(pred - label))
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
    # optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

    # loss summary
    loss_sum = tf.summary.scalar("loss", cost)
    summary_writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())

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

    # error = []

    # Iterate over training steps.
    for step in range(NUM_STEPS):
        data_id = step * BATCH_SIZE % TOTAL_DATA
        x_, y_ = load_travel_data(data_id, BATCH_SIZE)

        if step % SAVE_PRED_EVERY == 0:
            summary, _ = sess.run([loss_sum, optimizer], feed_dict={data: x_, label: y_})
            summary_writer.add_summary(summary, step)
            save(saver, sess, SNAPSHOT_DIR, step)
        else:
            summary_str, _ = sess.run([loss_sum, optimizer], feed_dict={data: x_, label: y_})
            summary_writer.add_summary(summary_str, step)
        if step % 3000 == 0:
            y, res = sess.run([label, pred], feed_dict={data: x_, label: y_})
            print('step {:d} \t y = {:.3f}, res = {:.3f}, loss = {:.3f}'.format(step, y[0][0], res[0][0], np.abs(y[0][0]-res[0][0])/y[0][0]))
        
    #     y, res = sess.run([label, pred], feed_dict={data: x_, label: y_})
    #     error.append(np.abs(y[0][0]-res[0][0])/y[0][0])
    # print np.mean(error)



if __name__ == '__main__':

    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        main(sess)




