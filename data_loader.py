#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup
from nltk.tokenize.toktok import ToktokTokenizer
import re
import os

nltk.download('stopwords')
tokenizer=ToktokTokenizer()

def review_to_words(text):
    soup = BeautifulSoup(text,"html.parser")
    text=soup.get_text()
    text = re.sub('\[[^]]*\]', '', text)
    text = re.sub(r"[^a-zA-Z]"," ",text)
    tokens = tokenizer.tokenize(text.lower())
    tokens = [token.strip() for token in tokens]
    ps=PorterStemmer()
    stemmed_tokens = [ps.stem(word) for word in tokens]
    stop=set(stopwords.words('english'))
    filtered_tokens = [token for token in stemmed_tokens if token not in stop]
    return ' '.join(filtered_tokens)


def get_data():
    filenames = []
    for _,_,file in os.walk('/home/pulkit/Desktop/aclImdb/train/pos'):
        filenames = file
    x_train=[]
    y_train=[]   
    for filename in filenames:
         with open('/home/pulkit/Desktop/aclImdb/train/pos/'+filename, 'r') as f:
             corpus = f.read()
             x_train.append(corpus)
             y_train.append(int(filename[-5]))
                    
    for _,_,file in os.walk('/home/pulkit/Desktop/aclImdb/train/neg'):
        filenames = file
    for filename in filenames:
         with open('/home/pulkit/Desktop/aclImdb/train/neg/'+filename, 'r') as f:
             corpus = f.read()
             x_train.append(corpus)
             y_train.append(int(filename[-5]))
    return x_train,y_train


