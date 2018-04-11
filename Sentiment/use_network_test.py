# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 17:35:20 2018

@author: bsingh46
"""

import tensorflow as tf
from  tf_text import create_feature_set_and_labels
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter
import re
from nltk.corpus import stopwords
lemmatizer=WordNetLemmatizer()
from tf_train import neural_network_model

train_x,train_y,test_x, test_y=create_feature_set_and_labels('negative.txt','positive.txt')
model_path = "model.ckpt"
x=tf.placeholder('float',[None,len(train_x[0])])
                
epochs=10

def use_neural_network(input_data):
    #saver = tf.train.import_meta_graph('model.ckpt.meta')
    prediction=neural_network_model(x)
    #print(prediction)
    
    lexicon=[]
    with open('lexicon.pickle','rb') as f:
        lexicon=pickle.load(f)
    #print((lexicon))
        
        
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('model/model.ckpt.meta')
        
        sess.run(tf.initialize_all_variables())
        for epoch in range(epochs):
            try:
                new_saver.restore(sess, tf.train.latest_checkpoint('model/'))
            except Exception as e:
                print(str(e))
            #epoch_loss=0
            
            
        current_words=word_tokenize(input_data.lower())
        current_words=[lemmatizer.lemmatize(i) for i in current_words]
        
        features=np.zeros(len(lexicon))
        
        for word in current_words:
            if word.lower() in lexicon:
                index_value=lexicon.index(word.lower())
                features[index_value]+=1
                
        features=np.array(list(features))
        #print(features)
        
        
        ### load the saved session
        result=sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}),1))
        print(prediction.eval(feed_dict={x:[features]}))
        
        if result[0]==0:
            print("Negative:   ", input_data)
            return(1)
        elif result[0]==1:
            print("Positive:   ", input_data)
            return(0)

  
            
            
        
        
use_neural_network("BEAUTIFUL SUNNY DAY IN PARK")   
use_neural_network("What a terrible weather")
    
    






# =============================================================================
# 
# #### why do we use softmax
#     
# If we take an input of [1, 2, 3, 4, 1, 2, 3], the softmax of that is [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]. The output has most of its weight where the '4' was in the original input. This is what the function is normally used for: to highlight the largest values and suppress values which are significantly below the maximum value. But note: softmax is not scale invariant, so if the input were [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3] (which sums to 1.6) the softmax would be [0.125, 0.138, 0.153, 0.169, 0.125, 0.138, 0.153]. This shows that for values between 0 and 1 softmax in fact de-emphasizes the maximum value (note that 0.169 is not only less than 0.475, it is also less than the initial value of 0.4).
# 
# Computation of this example using simple Python code:
# 
# >>> import math
# >>> z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
# >>> z_exp = [math.exp(i) for i in z]
# >>> print([round(i, 2) for i in z_exp])
# [2.72, 7.39, 20.09, 54.6, 2.72, 7.39, 20.09]
# >>> sum_z_exp = sum(z_exp)
# >>> print(round(sum_z_exp, 2))
# 114.98
# >>> softmax = [round(i / sum_z_exp, 3) for i in z_exp]
# >>> print(softmax)
# [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]
# 
# 
# 
# =============================================================================






































