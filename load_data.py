import numpy as np


def load_train_travel_data(batch_files):
    x_ = []
    y_ = []
    for id_ in xrange(len(batch_files)):
        filename = 'dataset/dataSets/training_travel_data/travel_{}.txt'.format(batch_files[id_][:-1])
        with open(filename, 'r') as fr:
            line_ = fr.readline()
        data_ = line_.split(' ')
        item_x = [float(tt) for tt in data_]
        item_y = [item_x[7]]
        del item_x[7]
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




def load_train_travel_data_six(idx, BATCH_SIZE):
    x_ = []
    y_ = []
    ID_FILE = 'dataset/dataSets/train_6_id.txt'
    with open(ID_FILE, 'r') as list_file:
        data_list = list_file.readlines()
    batch_files = data_list[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
    for id_ in xrange(len(batch_files)):
        filename = 'dataset/dataSets/multi_train_data/travel_6_{}.txt'.format(batch_files[id_][:-1])
        with open(filename, 'r') as fr:
            line_ = fr.readline()
        data_ = line_.split(' ')
        item_x = [float(tt) for tt in data_]
        item_y = [item_x[7]]
        del item_x[7]
        x_.append(item_x)
        y_.append(item_y)
    return np.array(x_), np.array(y_)

def load_train_travel_data_five(idx, BATCH_SIZE):
    x_ = []
    y_ = []
    ID_FILE = 'dataset/dataSets/train_5_id.txt'
    with open(ID_FILE, 'r') as list_file:
        data_list = list_file.readlines()
    batch_files = data_list[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
    for id_ in xrange(len(batch_files)):
        filename = 'dataset/dataSets/multi_train_data/travel_5_{}.txt'.format(batch_files[id_][:-1])
        with open(filename, 'r') as fr:
            line_ = fr.readline()
        data_ = line_.split(' ')
        item_x = [float(tt) for tt in data_]
        item_y = [item_x[6]]
        del item_x[6]
        x_.append(item_x)
        y_.append(item_y)
    return np.array(x_), np.array(y_)

def load_train_travel_data_four(idx, BATCH_SIZE):
    x_ = []
    y_ = []
    ID_FILE = 'dataset/dataSets/train_4_id.txt'
    with open(ID_FILE, 'r') as list_file:
        data_list = list_file.readlines()
    batch_files = data_list[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
    for id_ in xrange(len(batch_files)):
        filename = 'dataset/dataSets/multi_train_data/travel_4_{}.txt'.format(batch_files[id_][:-1])
        with open(filename, 'r') as fr:
            line_ = fr.readline()
        data_ = line_.split(' ')
        item_x = [float(tt) for tt in data_]
        item_y = [item_x[5]]
        del item_x[5]
        x_.append(item_x)
        y_.append(item_y)
    return np.array(x_), np.array(y_)

def load_train_travel_data_three(idx, BATCH_SIZE):
    x_ = []
    y_ = []
    ID_FILE = 'dataset/dataSets/train_3_id.txt'
    with open(ID_FILE, 'r') as list_file:
        data_list = list_file.readlines()
    batch_files = data_list[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
    for id_ in xrange(len(batch_files)):
        filename = 'dataset/dataSets/multi_train_data/travel_3_{}.txt'.format(batch_files[id_][:-1])
        with open(filename, 'r') as fr:
            line_ = fr.readline()
        data_ = line_.split(' ')
        item_x = [float(tt) for tt in data_]
        item_y = [item_x[4]]
        del item_x[4]
        x_.append(item_x)
        y_.append(item_y)
    return np.array(x_), np.array(y_)

def load_train_travel_data_two(idx, BATCH_SIZE):
    x_ = []
    y_ = []
    ID_FILE = 'dataset/dataSets/train_2_id.txt'
    with open(ID_FILE, 'r') as list_file:
        data_list = list_file.readlines()
    batch_files = data_list[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
    for id_ in xrange(len(batch_files)):
        filename = 'dataset/dataSets/multi_train_data/travel_2_{}.txt'.format(batch_files[id_][:-1])
        with open(filename, 'r') as fr:
            line_ = fr.readline()
        data_ = line_.split(' ')
        item_x = [float(tt) for tt in data_]
        item_y = [item_x[3]]
        del item_x[3]
        x_.append(item_x)
        y_.append(item_y)
    return np.array(x_), np.array(y_)

def load_train_travel_data_one(idx, BATCH_SIZE):
    x_ = []
    y_ = []
    ID_FILE = 'dataset/dataSets/train_1_id.txt'
    with open(ID_FILE, 'r') as list_file:
        data_list = list_file.readlines()
    batch_files = data_list[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
    for id_ in xrange(len(batch_files)):
        filename = 'dataset/dataSets/multi_train_data/travel_1_{}.txt'.format(batch_files[id_][:-1])
        with open(filename, 'r') as fr:
            line_ = fr.readline()
        data_ = line_.split(' ')
        item_x = [float(tt) for tt in data_]
        item_y = [item_x[2]]
        del item_x[2]
        x_.append(item_x)
        y_.append(item_y)
    return np.array(x_), np.array(y_)
