# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 14:20:01 2018

@author: bsingh46

"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter
import re
from nltk.corpus import stopwords
#import enchant



### lemmatizer can give a memory error - ran outta ram
lemmatizer=WordNetLemmatizer()
hm_lines=100000000



def clean_str(string):

    string = re.sub(r"[^A-Za-z]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()






def create_lexicon(a, b):
    lexicon=[]
    for fi in [a, b]:
        with open(fi,'r',encoding="utf8",errors='ignore') as f:
            content=f.readlines()
            #print(content)
            for l in content[:hm_lines]:
                l=clean_str(l)
                all_words=word_tokenize(l)
                ##print(all_words)
                lexicon+=list(all_words)
    ## stemmer            
    lexicon=[lemmatizer.lemmatize(i) for i in lexicon]
    #lexicon=[w for w in lexicon if w in nltk.corpus.words.words()]
    ## gives a dictionary of word occurence {'the':7,"you":9}
    #print(lexicon)
    word_counts=Counter(lexicon)
    temp=[]
    ## dont want sparse words and very common words such as articles
    for w in word_counts:
        if word_counts[w] > 5:
            temp.append(w)
    stopset = set(stopwords.words('english'))    
    temp=[i for i in temp if i not in stopset]
    temp=[i for i in temp if len(i)>3]
    #temp=[w for w in temp if w in nltk.corpus.words.words()]
    #print(len(temp)) 
    #temp=[clean_str(i) for i in temp]
    #print(len(temp)) 
    #print(temp)   
    with open('lexicon.pickle','wb') as f:
        pickle.dump(temp,f)    
    return temp


## classifying feature sets
def sample_handling(sample,lexicon, classification):
    featureset=[]
    
    with open(sample,'r',encoding="utf8",errors='ignore') as f:
        content=f.readlines()
        for l in content[:hm_lines]:
            current_words=word_tokenize(l.lower())
            current_words=[lemmatizer.lemmatize(i) for i in current_words]
            ##print(current_words)
            features=np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    ##print(word)
                    index_value=lexicon.index(word.lower())
                    ##print(index_value)
                    features[index_value]+=1
            features=list(features)
            featureset.append([features,classification])
            ##[[[1 0 0 0 1 0 0 1],[0,1]],[[0 0 1 0 1 1],[1,0]]]
            ## [0,1] -> negative sentiment
            ## [1,0] -> positive sentiment
        
        return featureset
    
    
    
def create_feature_set_and_labels(a, b,test_size=0.1):
    lexicon=create_lexicon(a, b)
    features=[]
    features+=sample_handling('negative.txt',lexicon,[1,0])
    features+=sample_handling('positive.txt',lexicon,[0,1])

    ### we need to shuffle features for the neural network
    ## 
    random.shuffle(features)
    features=np.array(features)
    testing_size=int(test_size*len(features))
    ###[[features,label],[features,label]]
    ### feature[:,0] -> gives us the 0th element that is the features from the above array
    train_x=list(features[:,0][:-testing_size])
    train_y=list(features[:,1][:-testing_size])
    
    
    test_x=list(features[:,0][-testing_size:])
    test_y=list(features[:,1][-testing_size:])
    
    
    return train_x,train_y,test_x, test_y



##aes=551 ; crime=51 (); sde=3179; traffic=701; walk=148 (500)
if __name__=='__main__':
    train_x,train_y,test_x, test_y=create_feature_set_and_labels('negative.txt','positive.txt')
    with open('sentiment_set.pickle','wb') as f:
        pickle.dump([train_x,train_y,test_x,test_y],f)
        ### why we use pickle https://pythontips.com/2013/08/02/what-is-pickle-in-python/
        
        
        
        
        
# =============================================================================
# import string
# s = '... some string with punctuation ...'
# s = s.translate(None, string.punctuation)
# 
# import enchant
# d = enchant.Dict("en_US")
# d.check("Hello")
# d.check("Helo")
# d.suggest("Helo")
# =============================================================================
        
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            
            
            
    