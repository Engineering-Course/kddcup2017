import tensorflow as tf
import numpy as np
import os
import time
from load_data import *
from model_setting import *
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Parameters
BATCH_SIZE = 5
LEARNING_RATE = 5e-6
TRAINING_FLAG = True
EPOCH = 5000
SNAPSHOT_DIR = './checkpoint3'
LOG_DIR = './logs/three'
TRAIN_ID_FILE = 'dataset/dataSets/train_3_id.txt'
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
    # data6 = tf.placeholder(tf.float32, [BATCH_SIZE, six_input], name='data6')
    # data5 = tf.placeholder(tf.float32, [BATCH_SIZE, five_input], name='data5')
    # data4 = tf.placeholder(tf.float32, [BATCH_SIZE, four_input], name='data4')
    data3 = tf.placeholder(tf.float32, [BATCH_SIZE, three_input], name='data3')
    # data2 = tf.placeholder(tf.float32, [BATCH_SIZE, two_input], name='data2')
    # data1 = tf.placeholder(tf.float32, [BATCH_SIZE, one_input], name='data1')
    # label6 = tf.placeholder(tf.float32, [BATCH_SIZE, n_classes], name='label6')
    # label5 = tf.placeholder(tf.float32, [BATCH_SIZE, n_classes], name='label5')
    # label4= tf.placeholder(tf.float32, [BATCH_SIZE, n_classes], name='label4')
    label3 = tf.placeholder(tf.float32, [BATCH_SIZE, n_classes], name='label3')
    # label2 = tf.placeholder(tf.float32, [BATCH_SIZE, n_classes], name='label2')
    # label1 = tf.placeholder(tf.float32, [BATCH_SIZE, n_classes], name='label1')

    # pred6 = net_six(data6, 'six')
    # pred5 = net_five(data5, 'five')
    # pred4 = net_four(data4, 'four')
    pred3 = net_three(data3, 'three')
    # pred2 = net_two(data2, 'two')
    # pred1 = net_one(data1, 'one')

    # Define loss and optimizer
    # cost6 = tf.reduce_mean(tf.abs(pred6 - label6))
    # cost5 = tf.reduce_mean(tf.abs(pred5 - label5))
    # cost4 = tf.reduce_mean(tf.abs(pred4 - label4))
    cost3 = tf.reduce_mean(tf.abs(pred3 - label3))
    # cost2 = tf.reduce_mean(tf.abs(pred2 - label2))
    # cost1 = tf.reduce_mean(tf.abs(pred1 - label1))
    # cost = (cost6 + cost5 + cost4 + cost3 + cost2 + cost1) / 1.0
    cost = cost3
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

    if TRAINING_FLAG:
        # Iterate over training steps.
        counter = 1
        fi = open('record3.txt', 'w')
        for ee in xrange(EPOCH):
            with open(TRAIN_ID_FILE, 'r') as list_file:
                data_list = list_file.readlines()
            np.random.shuffle(data_list)
            batch_idxs = len(data_list) // BATCH_SIZE
            

            for idx in xrange(0, batch_idxs):
                # x_6, y_6 = load_train_travel_data_six(idx, BATCH_SIZE)
                # x_5, y_5 = load_train_travel_data_five(idx, BATCH_SIZE)
                # x_4, y_4 = load_train_travel_data_four(idx, BATCH_SIZE)
                x_3, y_3 = load_train_travel_data_three(idx, BATCH_SIZE)
                # x_2, y_2 = load_train_travel_data_two(idx, BATCH_SIZE)
                # x_1, y_1 = load_train_travel_data_one(idx, BATCH_SIZE)

                summary, _ = sess.run([loss_sum, optimizer], feed_dict={data3: x_3, label3: y_3})
                summary_writer.add_summary(summary, counter)
                counter += 1

            save(saver, sess, SNAPSHOT_DIR, counter)
            ##  Test...
            # error6 = []
            # error5 = []
            # error4 = []
            error3 = []
            # error2 = []
            # error1 = []
            for idx in xrange(0, batch_idxs):
                # x_6, y_6 = load_train_travel_data_six(idx, BATCH_SIZE)
                # x_5, y_5 = load_train_travel_data_five(idx, BATCH_SIZE)
                # x_4, y_4 = load_train_travel_data_four(idx, BATCH_SIZE)
                x_3, y_3 = load_train_travel_data_three(idx, BATCH_SIZE)
                # x_2, y_2 = load_train_travel_data_two(idx, BATCH_SIZE)
                # x_1, y_1 = load_train_travel_data_one(idx, BATCH_SIZE)

                # y6, res6 = sess.run([label6, pred6], feed_dict={data6: x_6, label6: y_6})
                # y5, res5 = sess.run([label5, pred5], feed_dict={data5: x_5, label5: y_5})
                # y4, res4 = sess.run([label4, pred4], feed_dict={data4: x_4, label4: y_4})
                y3, res3 = sess.run([label3, pred3], feed_dict={data3: x_3, label3: y_3})
                # y2, res2 = sess.run([label2, pred2], feed_dict={data2: x_2, label2: y_2})
                # y1, res1 = sess.run([label1, pred1], feed_dict={data1: x_1, label1: y_1})
                for bb in xrange(BATCH_SIZE):
                    # error6.append(np.abs(y6[bb][0]-res6[bb][0])/y6[bb][0])
                    # error5.append(np.abs(y5[bb][0]-res5[bb][0])/y5[bb][0])
                    # error4.append(np.abs(y4[bb][0]-res4[bb][0])/y4[bb][0])
                    error3.append(np.abs(y3[bb][0]-res3[bb][0])/y3[bb][0])
                    # error2.append(np.abs(y2[bb][0]-res2[bb][0])/y2[bb][0])
                    # error1.append(np.abs(y1[bb][0]-res1[bb][0])/y1[bb][0])
            # test_res6 = np.mean(error6)
            # test_res5 = np.mean(error5)
            # test_res4 = np.mean(error4)
            test_res3 = np.mean(error3)
            # test_res2 = np.mean(error2)
            # test_res1 = np.mean(error1)
            # ave_res = np.mean([test_res6, test_res5, test_res4, test_res3, test_res2, test_res1])
            print('test_res: {:f}, epoch {:f}, step {:d}'.format(test_res3, ee, counter))
            fi.write('test_res: {:f}, epoch {:f}, step {:d}\n'.format(test_res3, ee, counter))
            # print('epoch {:f}, step {:d}'.format(ee, counter))
            # print('ave: {:f}, six: {:f}, five: {:f}, four: {:f}, three: {:f}, two: {:f}, one: {:f}'.format(ave_res, test_res6, test_res5, test_res4, test_res3, test_res2, test_res1))
            # fi.write('ave: {:f}, six: {:f}, five: {:f}, four: {:f}, three: {:f}, two: {:f}, one: {:f}'.format(ave_res, test_res6, test_res5, test_res4, test_res3, test_res2, test_res1))
        fi.close()
    else:
        # error6 = []
        # error5 = []
        # error4 = []
        error3 = []
        # error2 = []
        # error1 = []
        with open(TRAIN_ID_FILE, 'r') as list_file:
            data_list = list_file.readlines()
        batch_idxs = len(data_list) // BATCH_SIZE

        for idx in xrange(0, batch_idxs):
            # x_6, y_6 = load_train_travel_data_six(idx, BATCH_SIZE)
            # x_5, y_5 = load_train_travel_data_five(idx, BATCH_SIZE)
            # x_4, y_4 = load_train_travel_data_four(idx, BATCH_SIZE)
            x_3, y_3 = load_train_travel_data_three(idx, BATCH_SIZE)
            # x_2, y_2 = load_train_travel_data_two(idx, BATCH_SIZE)
            # x_1, y_1 = load_train_travel_data_one(idx, BATCH_SIZE)

            # y6, res6 = sess.run([label6, pred6], feed_dict={data6: x_6, label6: y_6})
            # y5, res5 = sess.run([label5, pred5], feed_dict={data5: x_5, label5: y_5})
            # y4, res4 = sess.run([label4, pred4], feed_dict={data4: x_4, label4: y_4})
            y3, res3 = sess.run([label3, pred3], feed_dict={data3: x_3, label3: y_3})
            # y2, res2 = sess.run([label2, pred2], feed_dict={data2: x_2, label2: y_2})
            # y1, res1 = sess.run([label1, pred1], feed_dict={data1: x_1, label1: y_1})
            for bb in xrange(BATCH_SIZE):
                # error6.append(np.abs(y6[bb][0]-res6[bb][0])/y6[bb][0])
                # error5.append(np.abs(y5[bb][0]-res5[bb][0])/y5[bb][0])
                # error4.append(np.abs(y4[bb][0]-res4[bb][0])/y4[bb][0])
                error3.append(np.abs(y3[bb][0]-res3[bb][0])/y3[bb][0])
                # error2.append(np.abs(y2[bb][0]-res2[bb][0])/y2[bb][0])
                # error1.append(np.abs(y1[bb][0]-res1[bb][0])/y1[bb][0])
        # test_res6 = np.mean(error6)
        # test_res5 = np.mean(error5)
        # test_res4 = np.mean(error4)
        test_res3 = np.mean(error3)
        # test_res2 = np.mean(error2)
        # test_res1 = np.mean(error1)
        # print('six: {:f}, five: {:f}, four: {:f}, three: {:f}, two: {:f}, one: {:f}').format(test_res6, test_res5, test_res4, test_res3, test_res2, test_res1)
        print('test_res: {:f}'.format(test_res3))



if __name__ == '__main__':

    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        main(sess)




