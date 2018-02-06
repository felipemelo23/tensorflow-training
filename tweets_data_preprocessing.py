#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 10:12:23 2018

@author: Felipe Melo

Some notes about the dataset:
- polarity 0 = negative. 2 = neutral. 4 = positive.
- id
- date
- query
- user
- tweet
"""

import nltk
import pickle
import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def init_process(fin, fout):
    outfile = open(fout, 'a')
    with open(fin, buffering=200000, encoding='latin-1') as file:
        try:
            for line in file:
                line = line.replace('"','')
                initial_polarity = line.split(',')[0]
                
                if initial_polarity == '0':
                    initial_polarity = [1,0]
                elif initial_polarity == '4':
                    initial_polarity = [0,1]
                    
                tweet = line.split(',')[-1]
                outline = str(initial_polarity) + ':::' + tweet
                outfile.write(outline)
        except Exception as e:
            print(str(e))
    outfile.close()
                               
def create_lexicon(fin):
    lexicon = []
    with open(fin, 'r', buffering=100000, encoding='latin-1') as f:
        try:
            counter = 1
            content = ''
            for line in f:
                counter += 1
                if (counter/2500.0).is_integer():
                    tweet = line.split(':::')[1]
                    content += ' '+tweet
                    words = word_tokenize(content)
                    words = [lemmatizer.lemmatize(i) for i in words]
                    lexicon = list(set(lexicon + words))
                    print(counter, len(lexicon))

        except Exception as e:
            print(str(e))

    with open('data/twitter_lexicon.pickle','wb') as f:
        pickle.dump(lexicon,f) 
        
def create_test_data_pickle(fin):

    feature_sets = []
    labels = []
    counter = 0
    with open(fin, 'r', buffering=20000, encoding='latin-1') as f:
        for line in f:
            try:
                splitted_line = line.split(':::')
                features = list(eval(splitted_line[0]))
                label = splitted_line[1]
                feature_sets.append(features)
                labels.append(label)
                counter += 1
            except:
                pass
    print(counter)
    feature_sets = np.array(feature_sets)
    labels = np.array(labels)



#init_process('data/training.1600000.processed.noemoticon.csv','data/train_set.csv')
#init_process('data/testdata.manual.2009.06.14.csv','data/test_set.csv')  
#create_lexicon('data/train_set.csv')
create_test_data_pickle('data/test_set.csv')         
                
                
                    