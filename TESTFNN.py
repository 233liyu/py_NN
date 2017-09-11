#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import pandas as pd
import tpyeexchange
#from sklearn import cross_validation
from sklearn.model_selection import train_test_split
# Using Tensorflow's default tools to fetch data, this is the same as what we did in the first homework assignment.
#mnist = input_data.read_data_sets('./mnist', one_hot=True)

subtrainLabel = pd.read_csv('11/subtrainLabels.csv')
subtrainfeature = pd.read_csv("11/imgfeature.csv")
subtrain = pd.merge(subtrainLabel, subtrainfeature, on='Id')
labels = subtrain.Class
subtrain.drop(["Class", "Id"], axis=1, inplace=True)
subtrain = subtrain.as_matrix()
X_train, X_test, y_train, y_test = train_test_split(subtrain, labels, test_size=0.4)
X_train=X_train.__div__(255.0)
X_test=X_test.__div__(255.0)
y_train=tpyeexchange.changetpye1(y_train)
y_test=tpyeexchange.changetpye2(y_test)
print X_train
print y_train

def add_layer(inputs, in_size, out_size, activation_function=None,):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    #print (y_pre)
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1500]) # 28x28
ys = tf.placeholder(tf.float32, [None, 9])

# add output layer
prediction = add_layer(xs, 1500, 9,  activation_function=tf.nn.softmax)

# the error between prediction and real data
#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y, name="loss")
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.Session()

for i in range(1000):
    batch_xs=X_train
    batch_ys = y_train
    #print (batch_xs)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(
            X_test, y_test))

print(batch_ys)
