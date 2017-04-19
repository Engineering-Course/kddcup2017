import numpy as np


def load_travel_data(data_id, batch_size):
    x_ = []
    y_ = []
    for id_ in xrange(data_id, data_id + batch_size):
        filename = 'dataset/dataSets/travel_data2/travel_{}.txt'.format(id_)
        with open(filename, 'r') as fr:
            line_ = fr.readline()
        data_ = line_.split(' ')
        item_x = [float(tt) for tt in data_]
        item_y = [item_x[7]]
        del item_x[7]
        x_.append(item_x)
        y_.append(item_y)

    return np.array(x_), np.array(y_)

