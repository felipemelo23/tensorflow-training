#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 10:53:23 2018

@author: Felipe Melo
"""

import tensorflow as tf
import pickle
import numpy as np
import nltk

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class DenseLayer():
    def __init__(self, n_nodes=100, entries, outputs, activation=tf.nn.relu):
        self.n = n_nodes
        self.w = tf.Variable(tf.random_normal([entries,outputs]))
        self.b = tf.Variable(tf.random_normal([outputs]))
        self.activation = activation
        
    def flow(X):
        return self.act_function(tf.add(tf.matmul(X,self.w), self.b))

class NeuralNetwork():
    def __init__(self):
        self.layers = []
        
    def add_layer(self, layer):
        self.layers.append(layer)
        
    def set_the_flow(data):
        tensor = data
        
        for layer in self.layers:
            tensor = layer.flow(tensor)
        
        return tensor

    def set_up(X,y):
        pred = self.set_the_flow(X)
        
        ''' TO DO - Training the Model '''        

lemmatizer = WordNetLemmatizer()

batch_size = 32
total_batches = int(1600000/batch_size)

epochs = 10

X = tf.placeholder('float')
y = tf.placeholder('float')

saver = tf.train.Saver()
tf_log = 'tf.log'




