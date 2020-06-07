#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  jan  7 16:51:06 2020

@author: Anvesh
"""

import pandas as pd

dataset= read_csv('train.csv')

df.head()

dataset=dataset.dropna()

#get the independednt features

X=dataset.drop('label',axis=1)

y=dataset['label']

X.shape()
y.shape()

import tensorflow as tf

from tf.keras.layers import embedding
from tf.keras.preprocessing.sequence import pad_sequences
from tf.keras.models import sequential
from tf.keras.preprocessing.text import one-hot
from tf.keras.layers imort LSTM
from tf.keras.layers import dense

voc-size= 10000

#one hot representation

messages=x.copy()
messages.reset_index(inplase=true)

import nltk
import re
from nltk.corpus import stopwords
nltk.download('stop words')


### data pre processing

from nltk.stem.porter import porterstemmer
ps=porterstemmer()
corpus=[]
for i in range (0,len(messages)):
    print(i)
    review=re.sub('[^a-zA-Z]','',messages['title'][i])
    review=review.lower()
    review= review.split()
    
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    
    review=''.join(review)
    corpus.append(review)


##### one hot representation
    onehot_repr= [one_hot(words,voc_size)for words in corpus]
    
##### embedding layer pad sequence
    sent_length=20
    embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
    
    len(embedded_docs)
    
    ### creating model
    embedding_vector_feature=40
    model= seequential()
    model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
    model.add(lstm(100))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    print(model.summary)







