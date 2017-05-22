import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops

n_classes = 1

six_hidden_1 = 8 # 1st layer number of features
six_hidden_2 = 8 # 2nd layer number of features
six_hidden_3 = 16
six_input = 8

def net_six(data, name):
  with tf.variable_scope(name) as scope:
    w1 = tf.Variable(tf.random_normal([six_input, six_hidden_1], stddev=0.1))
    b1 = tf.Variable(tf.random_normal([six_hidden_1], stddev=0.1))
    layer_1 = tf.add(tf.matmul(data, w1), b1)
    layer_1 = tf.nn.relu(layer_1)

    w2 = tf.Variable(tf.random_normal([six_hidden_1, six_hidden_2], stddev=0.1))
    b2 = tf.Variable(tf.random_normal([six_hidden_2], stddev=0.1))
    layer_2 = tf.add(tf.matmul(layer_1, w2), b2)
    layer_2 = tf.nn.relu(layer_2)

    w3 = tf.Variable(tf.random_normal([six_hidden_2, six_hidden_3], stddev=0.1))
    b3 = tf.Variable(tf.random_normal([six_hidden_3], stddev=0.1))
    layer_3 = tf.add(tf.matmul(layer_2, w3), b3)
    layer_3 = tf.nn.relu(layer_3)

    w_out = tf.Variable(tf.random_normal([six_hidden_3, n_classes], stddev=0.1))
    b_out = tf.Variable(tf.random_normal([n_classes], stddev=0.1))
    out_layer = tf.matmul(layer_3, w_out) + b_out
    return out_layer

#####################################################################################
five_hidden_1 = 7
five_hidden_2 = 7
five_hidden_3 = 14
five_input = 7

def net_five(data, name):
  with tf.variable_scope(name) as scope:
    w1 = tf.Variable(tf.random_normal([five_input, five_hidden_1], stddev=0.1))
    b1 = tf.Variable(tf.random_normal([five_hidden_1], stddev=0.1))
    layer_1 = tf.add(tf.matmul(data, w1), b1)
    layer_1 = tf.nn.relu(layer_1)

    w2 = tf.Variable(tf.random_normal([five_hidden_1, five_hidden_2], stddev=0.1))
    b2 = tf.Variable(tf.random_normal([five_hidden_2], stddev=0.1))
    layer_2 = tf.add(tf.matmul(layer_1, w2), b2)
    layer_2 = tf.nn.relu(layer_2)

    w3 = tf.Variable(tf.random_normal([five_hidden_2, five_hidden_3], stddev=0.1))
    b3 = tf.Variable(tf.random_normal([five_hidden_3], stddev=0.1))
    layer_3 = tf.add(tf.matmul(layer_2, w3), b3)
    layer_3 = tf.nn.relu(layer_3)

    w_out = tf.Variable(tf.random_normal([five_hidden_3, n_classes], stddev=0.1))
    b_out = tf.Variable(tf.random_normal([n_classes], stddev=0.1))
    out_layer = tf.matmul(layer_3, w_out) + b_out
    return out_layer

######################################################################################
four_hidden_1 = 6
four_hidden_2 = 6
four_hidden_3 = 12
four_input = 6

def net_four(data, name):
  with tf.variable_scope(name) as scope:
    w1 = tf.Variable(tf.random_normal([four_input, four_hidden_1], stddev=0.1))
    b1 = tf.Variable(tf.random_normal([four_hidden_1], stddev=0.1))
    layer_1 = tf.add(tf.matmul(data, w1), b1)
    layer_1 = tf.nn.relu(layer_1)

    w2 = tf.Variable(tf.random_normal([four_hidden_1, four_hidden_2], stddev=0.1))
    b2 = tf.Variable(tf.random_normal([four_hidden_2], stddev=0.1))
    layer_2 = tf.add(tf.matmul(layer_1, w2), b2)
    layer_2 = tf.nn.relu(layer_2)

    w3 = tf.Variable(tf.random_normal([four_hidden_2, four_hidden_3], stddev=0.1))
    b3 = tf.Variable(tf.random_normal([four_hidden_3], stddev=0.1))
    layer_3 = tf.add(tf.matmul(layer_2, w3), b3)
    layer_3 = tf.nn.relu(layer_3)

    w_out = tf.Variable(tf.random_normal([four_hidden_3, n_classes], stddev=0.1))
    b_out = tf.Variable(tf.random_normal([n_classes], stddev=0.1))
    out_layer = tf.matmul(layer_3, w_out) + b_out
    return out_layer

######################################################################################
three_hidden_1 = 5
three_hidden_2 = 5
three_hidden_3 = 10
three_input = 5

def net_three(data, name):
  with tf.variable_scope(name) as scope:
    w1 = tf.Variable(tf.random_normal([three_input, three_hidden_1], stddev=0.1))
    b1 = tf.Variable(tf.random_normal([three_hidden_1], stddev=0.1))
    layer_1 = tf.add(tf.matmul(data, w1), b1)
    layer_1 = tf.nn.relu(layer_1)

    w2 = tf.Variable(tf.random_normal([three_hidden_1, three_hidden_2], stddev=0.1))
    b2 = tf.Variable(tf.random_normal([three_hidden_2], stddev=0.1))
    layer_2 = tf.add(tf.matmul(layer_1, w2), b2)
    layer_2 = tf.nn.relu(layer_2)

    w3 = tf.Variable(tf.random_normal([three_hidden_2, three_hidden_3], stddev=0.1))
    b3 = tf.Variable(tf.random_normal([three_hidden_3], stddev=0.1))
    layer_3 = tf.add(tf.matmul(layer_2, w3), b3)
    layer_3 = tf.nn.relu(layer_3)

    w_out = tf.Variable(tf.random_normal([three_hidden_3, n_classes], stddev=0.1))
    b_out = tf.Variable(tf.random_normal([n_classes], stddev=0.1))
    out_layer = tf.matmul(layer_3, w_out) + b_out
    return out_layer

###########################################################################################
two_hidden_1 = 4
two_hidden_2 = 4
two_hidden_3 = 8
two_input = 4

def net_two(data, name):
  with tf.variable_scope(name) as scope:
    w1 = tf.Variable(tf.random_normal([two_input, two_hidden_1], stddev=0.1))
    b1 = tf.Variable(tf.random_normal([two_hidden_1], stddev=0.1))
    layer_1 = tf.add(tf.matmul(data, w1), b1)
    layer_1 = tf.nn.relu(layer_1)

    w2 = tf.Variable(tf.random_normal([two_hidden_1, two_hidden_2], stddev=0.1))
    b2 = tf.Variable(tf.random_normal([two_hidden_2], stddev=0.1))
    layer_2 = tf.add(tf.matmul(layer_1, w2), b2)
    layer_2 = tf.nn.relu(layer_2)

    w3 = tf.Variable(tf.random_normal([two_hidden_2, two_hidden_3], stddev=0.1))
    b3 = tf.Variable(tf.random_normal([two_hidden_3], stddev=0.1))
    layer_3 = tf.add(tf.matmul(layer_2, w3), b3)
    layer_3 = tf.nn.relu(layer_3)

    w_out = tf.Variable(tf.random_normal([two_hidden_3, n_classes], stddev=0.1))
    b_out = tf.Variable(tf.random_normal([n_classes], stddev=0.1))
    out_layer = tf.matmul(layer_3, w_out) + b_out
    return out_layer