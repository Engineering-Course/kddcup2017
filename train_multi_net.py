import tensorflow as tf
import numpy as np
import os
import time
from load_data import *
from model_setting import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Parameters
BATCH_SIZE = 5
LEARNING_RATE = 1e-5
TRAINING_FLAG = True
EPOCH = 500
SNAPSHOT_DIR = './checkpoint'
LOG_DIR = './logs'
TRAIN_ID_FILE = 'dataset/dataSets/train_id.txt'
TEST_ID_FIEL = 'dataset/dataSets/test_id.txt'

# Network Parameters
six_input = 8
five_input = 7
four_input = 6
three_input = 5
two_input = 4
one_input = 3
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
    data5 = tf.placeholder(tf.float32, [BATCH_SIZE, five_input], name='data5')
    data4 = tf.placeholder(tf.float32, [BATCH_SIZE, four_input], name='data4')
    data3 = tf.placeholder(tf.float32, [BATCH_SIZE, three_input], name='data3')
    data2 = tf.placeholder(tf.float32, [BATCH_SIZE, two_input], name='data2')
    data1 = tf.placeholder(tf.float32, [BATCH_SIZE, one_input], name='data1')
    label6 = tf.placeholder(tf.float32, [BATCH_SIZE, n_classes], name='label6')
    label5 = tf.placeholder(tf.float32, [BATCH_SIZE, n_classes], name='label5')
    label4= tf.placeholder(tf.float32, [BATCH_SIZE, n_classes], name='label4')
    label3 = tf.placeholder(tf.float32, [BATCH_SIZE, n_classes], name='label3')
    label2 = tf.placeholder(tf.float32, [BATCH_SIZE, n_classes], name='label2')
    label1 = tf.placeholder(tf.float32, [BATCH_SIZE, n_classes], name='label1')

    pred6 = net_six(data6, 'six')
    pred5 = net_five(data5, 'five')
    pred4 = net_four(data4, 'four')
    pred3 = net_three(data3, 'three')
    pred2 = net_two(data2, 'two')
    pred1 = net_one(data1, 'one')

    # Define loss and optimizer
    cost6 = tf.reduce_mean(tf.abs(pred6 - label6))
    cost5 = tf.reduce_mean(tf.abs(pred5 - label5))
    cost4 = tf.reduce_mean(tf.abs(pred4 - label4))
    cost3 = tf.reduce_mean(tf.abs(pred3 - label3))
    cost2 = tf.reduce_mean(tf.abs(pred2 - label2))
    cost1 = tf.reduce_mean(tf.abs(pred1 - label1))
    cost = cost6 + cost5 + cost4 + cost3 + cost2 + cost1
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

    # loss summary
    loss_six = tf.summary.scalar("loss6", cost6)
    loss_five = tf.summary.scalar("loss5", cost5)
    loss_four = tf.summary.scalar("loss4", cost4)
    loss_three = tf.summary.scalar("loss3", cost3)
    loss_two = tf.summary.scalar("loss2", cost2)
    loss_one = tf.summary.scalar("loss1", cost1)
    loss = tf.summary.scalar("loss", cost)
    loss_sum = tf.summary.merge([loss, loss_six, loss_five, loss_four, loss_three, loss_two, loss_one])
    summary_writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())

    tf.global_variables_initializer().run()

    # Saver for storing checkpoints of the model.
    # Restore variables
    restore_var = tf.global_variables()
    saver = tf.train.Saver(var_list=restore_var, max_to_keep=2000)
    loader = tf.train.Saver(var_list=restore_var)
    if load(loader, sess, SNAPSHOT_DIR):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")    

    if TRAINING_FLAG:
        # Iterate over training steps.
        counter = 1
        fi = open('record.txt', 'w')
        for ee in xrange(EPOCH):
            with open(TRAIN_ID_FILE, 'r') as list_file:
                data_list = list_file.readlines()
            np.random.shuffle(data_list)
            batch_idxs = len(data_list) // BATCH_SIZE
            

            for idx in xrange(0, batch_idxs):
                batch_files = data_list[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
                x_6, y_6 = load_train_travel_data_six(batch_files)
                x_5, y_5 = load_train_travel_data_five(batch_files)
                x_4, y_4 = load_train_travel_data_four(batch_files)

                summary, _ = sess.run([loss_sum, optimizer], feed_dict={data: x_, label: y_})
                summary_writer.add_summary(summary, counter)
                counter += 1

            save(saver, sess, SNAPSHOT_DIR, counter)
            ##  Test...
            error = []
            for idx in xrange(0, batch_idxs):
                batch_files = data_list[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
                x_, y_ = load_train_travel_data(batch_files)        
                y, res = sess.run([label, pred], feed_dict={data: x_, label: y_})
                for bb in xrange(BATCH_SIZE):
                    error.append(np.abs(y[bb][0]-res[bb][0])/y[bb][0])
            test_result = np.mean(error)

            y, res = sess.run([label, pred], feed_dict={data: x_, label: y_})
            print('test_result: {:f}, epoch {:f}, step {:d}, y = {:.3f}, res = {:.3f}, loss = {:.3f}'.format(test_result, ee, counter,  y[0][0], res[0][0], np.abs(y[0][0]-res[0][0])/y[0][0]))
            fi.write('test_result: {:f}, epoch {:f}, step {:d}, y = {:.3f}, res = {:.3f}, loss = {:.3f}\n'.format(test_result, ee, counter,  y[0][0], res[0][0], np.abs(y[0][0]-res[0][0])/y[0][0]))
        fi.close()
    else:
        error = []
        with open(TRAIN_ID_FILE, 'r') as list_file:
            data_list = list_file.readlines()
        batch_idxs = len(data_list) // BATCH_SIZE

        for idx in xrange(0, batch_idxs):
            batch_files = data_list[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
            x_, y_ = load_train_travel_data(batch_files)        
            y, res = sess.run([label, pred], feed_dict={data: x_, label: y_})
            error.append(np.abs(y[0][0]-res[0][0])/y[0][0])
        print np.mean(error)



if __name__ == '__main__':

    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        main(sess)




