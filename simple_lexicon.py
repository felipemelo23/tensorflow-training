#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 09:09:59 2018

@author: Felipe Melo
"""

import numpy as np
import random
import pickle
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
lines = 100000

def create_lexicon(pos, neg):
    lexicon = []
    with open(pos, 'r') as file:
        contents = file.readlines()
        for line in contents[:lines]:
            lexicon += list(word_tokenize(line))
            
    with open(neg, 'r') as file:
        contents = file.readlines()
        for line in contents[:lines]:
            lexicon += list(word_tokenize(line))
            
    lexicon = [lemmatizer.lemmatize(word) for word in lexicon]
    word_freq = Counter(lexicon)
    lexicon = []
    for word in word_freq:
        if 1000 > word_freq[word] > 50:
            lexicon.append(word)
            
    print('Lexicon Size: {}'.format(len(lexicon)))
    return lexicon

def sample_handling(sample, lexicon, classification):
    featureset = []
    
    with open(sample, 'r') as file:
        contents = file.readlines()
        
        for line in contents[:lines]:
            current_words = word_tokenize(line.lower())
            current_words = [lemmatizer.lemmatize(word) for word in current_words]
            
            features = np.zeros(len(lexicon))
            
            for word in current_words:
                if word.lower() in lexicon:
                    index = lexicon.index(word.lower())
                    features[index] += 1
            
            features = list(features)
            featureset.append([features, classification])
            
    return featureset


def create_feature_sets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling('data/pos.txt', lexicon, [1,0])
    features += sample_handling('data/neg.txt', lexicon, [0,1])
    
    random.shuffle(features)
    features = np.array(features)
    
    data_split = int((1-test_size)*len(features))
    
    train_X = list(features[:,0][:-data_split])
    train_y = list(features[:,1][:-data_split])
    test_X = list(features[:,0][-data_split:])
    test_y = list(features[:,1][-data_split:])
    
    return train_X, train_y, test_X, test_y

if __name__ == '__main__':
    train_X, train_y, test_X, test_y = create_feature_sets_and_labels('data/pos.txt','data/neg.txt')
    with open('data/sentiment_set.pickle', 'wb') as file:
        pickle.dump([train_X, train_y, test_X, test_y], file)

