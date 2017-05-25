import tensorflow as tf
import numpy as np
import os

def load_train_travel_data(batch_files, window):
    x_ = []
    y_ = []
    for id_ in xrange(len(batch_files)):
        filename = 'dataset/dataSets/data_training/travel_{}_{}.txt'.format(window, batch_files[id_][:-1])
        with open(filename, 'r') as fr:
            line_ = fr.readline()
        data_ = line_.split(' ')
        item_x = [float(tt) for tt in data_]
        item_y = [item_x[window+1]]
        del item_x[window+1]
        x_.append(item_x)
        y_.append(item_y)

    return np.array(x_), np.array(y_)


def load_test_travel_data(batch_id):
    x_ = []
    filename = 'dataset/dataSets/testing_travel_data/travel_{}.txt'.format(batch_id)
    with open(filename, 'r') as fr:
        line_ = fr.readline()
    data_ = line_.split(' ')
    item_x = [float(tt) for tt in data_]
    x_.append(item_x)

    return np.array(x_)



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


#
#def load_train_travel_data_six(idx, BATCH_SIZE):
#    x_ = []
#    y_ = []
#    ID_FILE = 'dataset/dataSets/train_6_id.txt'
#    with open(ID_FILE, 'r') as list_file:
#        data_list = list_file.readlines()
#    batch_files = data_list[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
#    for id_ in xrange(len(batch_files)):
#        filename = 'dataset/dataSets/multi_train_data/travel_6_{}.txt'.format(batch_files[id_][:-1])
#        with open(filename, 'r') as fr:
#            line_ = fr.readline()
#        data_ = line_.split(' ')
#        item_x = [float(tt) for tt in data_]
#        item_y = [item_x[7]]
#        del item_x[7]
#        x_.append(item_x)
#        y_.append(item_y)
#    return np.array(x_), np.array(y_)
#
#def load_train_travel_data_five(idx, BATCH_SIZE):
#    x_ = []
#    y_ = []
#    ID_FILE = 'dataset/dataSets/train_5_id.txt'
#    with open(ID_FILE, 'r') as list_file:
#        data_list = list_file.readlines()
#    batch_files = data_list[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
#    for id_ in xrange(len(batch_files)):
#        filename = 'dataset/dataSets/multi_train_data/travel_5_{}.txt'.format(batch_files[id_][:-1])
#        with open(filename, 'r') as fr:
#            line_ = fr.readline()
#        data_ = line_.split(' ')
#        item_x = [float(tt) for tt in data_]
#        item_y = [item_x[6]]
#        del item_x[6]
#        x_.append(item_x)
#        y_.append(item_y)
#    return np.array(x_), np.array(y_)
#
#def load_train_travel_data_four(idx, BATCH_SIZE):
#    x_ = []
#    y_ = []
#    ID_FILE = 'dataset/dataSets/train_4_id.txt'
#    with open(ID_FILE, 'r') as list_file:
#        data_list = list_file.readlines()
#    batch_files = data_list[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
#    for id_ in xrange(len(batch_files)):
#        filename = 'dataset/dataSets/multi_train_data/travel_4_{}.txt'.format(batch_files[id_][:-1])
#        with open(filename, 'r') as fr:
#            line_ = fr.readline()
#        data_ = line_.split(' ')
#        item_x = [float(tt) for tt in data_]
#        item_y = [item_x[5]]
#        del item_x[5]
#        x_.append(item_x)
#        y_.append(item_y)
#    return np.array(x_), np.array(y_)
#
#def load_train_travel_data_three(idx, BATCH_SIZE):
#    x_ = []
#    y_ = []
#    ID_FILE = 'dataset/dataSets/train_3_id.txt'
#    with open(ID_FILE, 'r') as list_file:
#        data_list = list_file.readlines()
#    batch_files = data_list[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
#    for id_ in xrange(len(batch_files)):
#        filename = 'dataset/dataSets/multi_train_data/travel_3_{}.txt'.format(batch_files[id_][:-1])
#        with open(filename, 'r') as fr:
#            line_ = fr.readline()
#        data_ = line_.split(' ')
#        item_x = [float(tt) for tt in data_]
#        item_y = [item_x[4]]
#        del item_x[4]
#        x_.append(item_x)
#        y_.append(item_y)
#    return np.array(x_), np.array(y_)
#
#def load_train_travel_data_two(idx, BATCH_SIZE):
#    x_ = []
#    y_ = []
#    ID_FILE = 'dataset/dataSets/train_2_id.txt'
#    with open(ID_FILE, 'r') as list_file:
#        data_list = list_file.readlines()
#    batch_files = data_list[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
#    for id_ in xrange(len(batch_files)):
#        filename = 'dataset/dataSets/multi_train_data/travel_2_{}.txt'.format(batch_files[id_][:-1])
#        with open(filename, 'r') as fr:
#            line_ = fr.readline()
#        data_ = line_.split(' ')
#        item_x = [float(tt) for tt in data_]
#        item_y = [item_x[3]]
#        del item_x[3]
#        x_.append(item_x)
#        y_.append(item_y)
#    return np.array(x_), np.array(y_)
#
#def load_train_travel_data_one(idx, BATCH_SIZE):
#    x_ = []
#    y_ = []
#    ID_FILE = 'dataset/dataSets/train_1_id.txt'
#    with open(ID_FILE, 'r') as list_file:
#        data_list = list_file.readlines()
#    batch_files = data_list[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
#    for id_ in xrange(len(batch_files)):
#        filename = 'dataset/dataSets/multi_train_data/travel_1_{}.txt'.format(batch_files[id_][:-1])
#        with open(filename, 'r') as fr:
#            line_ = fr.readline()
#        data_ = line_.split(' ')
#        item_x = [float(tt) for tt in data_]
#        item_y = [item_x[2]]
#        del item_x[2]
#        x_.append(item_x)
#        y_.append(item_y)
#    return np.array(x_), np.array(y_)
