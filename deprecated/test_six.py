import tensorflow as tf
import numpy as np
import os
import time
from load_data import *
from model_setting import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Parameters
BATCH_SIZE = 5
LEARNING_RATE = 5e-6
TRAINING_FLAG = True
EPOCH = 5000
SNAPSHOT_DIR = './checkpoint6'
LOG_DIR = './logs/six'
TRAIN_ID_FILE = 'dataset/dataSets/train_6_id.txt'
TEST_ID_FIEL = 'dataset/dataSets/test_id.txt'

# Network Parameters
six_input = 9
five_input = 8
four_input = 7
three_input = 6
two_input = 5
one_input = 4
n_classes = 1


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
    #print('The checkpoint has been created.')

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
    data6 = tf.placeholder(tf.float32, [BATCH_SIZE, six_input], name='data6')
    # data5 = tf.placeholder(tf.float32, [BATCH_SIZE, five_input], name='data5')
    # data4 = tf.placeholder(tf.float32, [BATCH_SIZE, four_input], name='data4')
    # data3 = tf.placeholder(tf.float32, [BATCH_SIZE, three_input], name='data3')
    # data2 = tf.placeholder(tf.float32, [BATCH_SIZE, two_input], name='data2')
    # data1 = tf.placeholder(tf.float32, [BATCH_SIZE, one_input], name='data1')
    label6 = tf.placeholder(tf.float32, [BATCH_SIZE, n_classes], name='label6')
    # label5 = tf.placeholder(tf.float32, [BATCH_SIZE, n_classes], name='label5')
    # label4= tf.placeholder(tf.float32, [BATCH_SIZE, n_classes], name='label4')
    # label3 = tf.placeholder(tf.float32, [BATCH_SIZE, n_classes], name='label3')
    # label2 = tf.placeholder(tf.float32, [BATCH_SIZE, n_classes], name='label2')
    # label1 = tf.placeholder(tf.float32, [BATCH_SIZE, n_classes], name='label1')

    pred6 = net_six(data6, 'six')
    # pred5 = net_five(data5, 'five')
    # pred4 = net_four(data4, 'four')
    # pred3 = net_three(data3, 'three')
    # pred2 = net_two(data2, 'two')
    # pred1 = net_one(data1, 'one')

    # Define loss and optimizer
    cost6 = tf.reduce_mean(tf.abs(pred6 - label6))
    # cost5 = tf.reduce_mean(tf.abs(pred5 - label5))
    # cost4 = tf.reduce_mean(tf.abs(pred4 - label4))
    # cost3 = tf.reduce_mean(tf.abs(pred3 - label3))
    # cost2 = tf.reduce_mean(tf.abs(pred2 - label2))
    # cost1 = tf.reduce_mean(tf.abs(pred1 - label1))
    # cost = (cost6 + cost5 + cost4 + cost3 + cost2 + cost1) / 1.0
    cost = cost6
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

    # loss summary
    # loss_six = tf.summary.scalar("loss_6", cost6)
    # loss_five = tf.summary.scalar("loss_5", cost5)
    # loss_four = tf.summary.scalar("loss_4", cost4)
    # loss_three = tf.summary.scalar("loss_3", cost3)
    # loss_two = tf.summary.scalar("loss_2", cost2)
    # loss_one = tf.summary.scalar("loss_1", cost1)
    loss_sum = tf.summary.scalar("loss", cost)
    # loss_sum = tf.summary.merge([loss, loss_six, loss_five, loss_four, loss_three, loss_two, loss_one])
    summary_writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())

    tf.global_variables_initializer().run()

    # Saver for storing checkpoints of the model.
    # Restore variables
    restore_var = tf.global_variables()
    saver = tf.train.Saver(var_list=restore_var, max_to_keep=5000)
    loader = tf.train.Saver(var_list=restore_var)
    if load(loader, sess, SNAPSHOT_DIR):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")    

    for step in range(NUM_STEPS):
        x_ = load_test_travel_data(step)
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



if __name__ == '__main__':

    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        main(sess)




