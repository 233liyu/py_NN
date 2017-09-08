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
subtrain = pd.merge(subtrainLabel,subtrainfeature,on='Id')
labels = subtrain.Class
subtrain.drop(["Class","Id"], axis=1, inplace=True)
subtrain = subtrain.as_matrix()
X_train, X_test, y_train, y_test = train_test_split(subtrain,labels,test_size=0.4)

#y_train1=tf.zeros([540,9],tf.int32)
#print X_train
#y1=y_train.as_matrix()

y_train=tpyeexchange.changetpye1(y_train)
y_test=tpyeexchange.changetpye2(y_test)
#print yy
#y_train.tolist()
#list(y_train)
#a=np.array([0,0,0,0,0,0,0,0,0],*540)
#multi = [ [0] * 9 ] * 540
#print multi
#list(a)

print X_train.shape
#for i in range(len(y_train)):
#    y_trainnew[i] =

#for i in range(y_train.shape[0]):
#    y_train1[y_train[0],y_train[1]]=1

#with tf.Session():
#    print y_train1.eval()

#a = np.array(([3, 2, 1], [2, 5, 7], [4, 7, 8]))

#itemindex = np.argwhere(a == 7)

#print (itemindex)

#print a


# Random seed.
rseed = 42
batch_size = 540
lr = 1e-1
num_epochs = 55
num_hiddens = 500
num_train, num_feats = X_train.shape
num_test = X_test.shape[0]
num_classes = 9

# Placeholders that should be filled with training pairs (x, y). Use None to unspecify the first dimension
# for flexibility.
x = tf.placeholder(tf.float32, [None, num_feats], name="x")
y = tf.placeholder(tf.int32, [None, num_classes], name="y")

# Model weights.
u1 =  2.0 * np.sqrt(6.0 / (num_feats + num_hiddens))
w1 = tf.Variable(tf.random_uniform(shape=[num_feats, num_hiddens], minval=-u1, maxval=u1), name="w1")
b1 = tf.Variable(tf.zeros([num_hiddens]), name="b1")
u2 =  2.0 * np.sqrt(6.0 / (num_hiddens + num_classes))
w2 = tf.Variable(tf.random_uniform(shape=[num_hiddens, num_classes], minval=-u2, maxval=u2), name="w2")
b2 = tf.Variable(tf.zeros([num_classes]), name="b2")

# logits is the log-probablity of each classes.
hiddens = tf.nn.relu(tf.matmul(x, w1) + b1)
logits = tf.matmul(hiddens, w2) + b2

# Use TensorFlow's default implementation to compute the cross-entropy loss of classification.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y, name="loss")
loss = tf.reduce_mean(cross_entropy)

# Build prediction function.
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(y, 1))
# Need to cast the type of correct_preds to float32 in order to compute the average mean accuracy.
accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

# Use TensorFlow's default implementation for optimziation algorithm. Note that we can understand
# an optimization procedure as an OP (operator) as well.
optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

# Start training!
num_batches = num_train / batch_size
losses = []
train_accs, valid_accs = [], []
x_batch=[[]]
y_batch=[[]]
time_start = time.time()
with tf.Session() as sess:
    # Before evaluating the graph, we should initialize all the variables.
    sess.run(tf.global_variables_initializer())
    for i in xrange(num_epochs):
        # Each training epoch contains num_batches of parameter updates.
        total_loss = 0.0
        for j in xrange(num_batches):
            # Fetch next mini-batch of data using TensorFlow's default method.
            #x_batch = X_train[batch_size * j:batch_size * (j + 1), :]
            #y_batch = y_train[batch_size * j:batch_size * (j + 1), :]
            x_batch = X_train
            y_batch = y_train

            # Note that we also need to include optimizer into the list in order to update parameters, but we
            # don't need the return value of optimizer.
            _, loss_batch = sess.run([optimizer, loss], feed_dict = {x: x_batch, y: y_batch})
            total_loss += loss_batch
        # Compute training set and validation set accuracy after each epoch.
        train_acc = sess.run([accuracy], feed_dict = {x: X_train, y: y_train})
        valid_acc = sess.run([accuracy], feed_dict = {x: X_test, y: y_test})
        losses.append(total_loss)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)
        print "Number of iteration: {}, total_loss = {}, train accuracy = {}, test accuracy = {}".format(i, total_loss, train_acc, valid_acc)
    # Evaluate the test set accuracy at the end.
    test_acc = sess.run([accuracy], feed_dict = {x: X_test, y: y_test})
time_end = time.time()
print "Time used for training = {} seconds.".format(time_end - time_start)
print "MNIST image classification accuracy on test set = {}".format(test_acc)

# Plot the losses during training.
plt.figure()
plt.title("MLP-1500-500-10 with TensorFlow")
plt.plot(losses, "b-o", linewidth=2)
plt.grid(True)
plt.xlabel("Iteration")
plt.ylabel("Cross-entropy")
plt.show()
