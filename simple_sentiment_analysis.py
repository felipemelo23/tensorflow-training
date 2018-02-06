#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 11:19:32 2018

@author: Felipe Melo
"""

import tensorflow as tf
import numpy as np
#from tensorflow.examples.tutorials.mnist import input_data
from simple_lexicon import create_feature_sets_and_labels

#mnist = input_data.read_data_sets('data', one_hot=True)
train_x,train_y,test_x,test_y = create_feature_sets_and_labels('data/pos.txt','data/neg.txt')

n_nodes_hlX = 1500

n_nodes_hl1 = n_nodes_hlX
n_nodes_hl2 = n_nodes_hlX
n_nodes_hl3 = n_nodes_hlX

n_classes = 2

batch_size = 100

x = tf.placeholder('float')
y = tf.placeholder('float')


def ann_model(data):
    # Building the structure of the network
    hl1 = {'w': tf.Variable(tf.random_normal([423, n_nodes_hl1])),
                      'b': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hl2 = {'w': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'b': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hl3 = {'w': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'b': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    ol = {'w': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'b': tf.Variable(tf.random_normal([n_classes]))}
    
    # Making it flow
    l1 = tf.add(tf.matmul(data, hl1['w']), hl1['b'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, hl2['w']), hl2['b'])
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.matmul(l2, hl3['w']), hl3['b'])
    l3 = tf.nn.relu(l3)
    
    output = tf.matmul(l3,ol['w']) + ol['b']
    
    return output

def ann_train(x):
    pred = ann_model(x)
    loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y) )
    
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    
    epochs = 50
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(epochs):
            epoch_loss = 0
            i=0
            while i < len(train_x):
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                
                _, l = sess.run([optimizer, loss], feed_dict={x: batch_x, y: batch_y})
                
                epoch_loss += l
                i += batch_size
            
            print('Epoch: {}/{} - loss: {}'.format(epoch+1, epochs, epoch_loss))
        
        correct = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: {}'.format(accuracy.eval({x: test_x, y: test_y})))
        
ann_train(x)