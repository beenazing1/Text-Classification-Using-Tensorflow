# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 17:35:20 2018

@author: bsingh46
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:10:34 2018

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

train_x,train_y,test_x, test_y=create_feature_set_and_labels('negative.txt','positive.txt')
model_path = "model/model.ckpt"



## define the model
## nodes for layers - 3 hidden layer

n_h1=5000
n_h2=5000
n_h3=5000


n_class=2
## goes through batches of a 100 tweets at a time
batch_size=25

tf.set_random_seed(1234)
x=tf.placeholder('float',[None,len(train_x[0])])
y=tf.placeholder('float')
w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')

### epochs
hm_epochs = 15
#
### formula : input_data*weight +biases
## if all of input data is 0 then no neoron is fired  - thus we include biases  - so that atleast some neuron gets fired even if all inputs are 0

hidden_1_layer={'weights':tf.Variable(tf.random_normal([len(train_x[0]),n_h1])),
                'biases':tf.Variable(tf.random_normal([n_h1]))}
hidden_2_layer={'weights':tf.Variable(tf.random_normal([n_h1,n_h2])),
                'biases':tf.Variable(tf.random_normal([n_h2]))}
hidden_3_layer={'weights':tf.Variable(tf.random_normal([n_h2,n_h3])),
                'biases':tf.Variable(tf.random_normal([n_h3]))}
output_layer={'weights':tf.Variable(tf.random_normal([n_h3,n_class])),
                'biases':tf.Variable(tf.random_normal([n_class]))}
    

def neural_network_model(data):
    
    

    ## model for each layer
    ### formula : input_data*weight +biases
    l1=tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])
    ## Tkes the value of l1 and passe sit through an actiavtion function
    l1=tf.nn.relu(l1)
    
    l2=tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases'])
    l2=tf.nn.relu(l2)
    
    l3=tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['biases'])
    l3=tf.nn.relu(l3)
    
    output2=tf.matmul(l3,output_layer['weights'])+output_layer['biases']
    print("output2:   ",output2)
    
    
# =============================================================================
    ## need to feed in a placeholder
#     with tf.Session() as sess:
#         output = sess.run(output2, feed_dict={x: l3, output_layer['weights'],output_layer['biases']}
#         print(output)
# =============================================================================
        
    return output2


saver=tf.train.Saver()

def train_neural_network(x):
    ## output is one hot array
    prediction=neural_network_model(x)    
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))  
    ### adam optimizer 
    ##https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
    optimizer=tf.train.AdamOptimizer().minimize(cost)
    init = tf.global_variables_initializer()
    #print(init)           
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)           
        epoch = 15
        epoch_loss = 1            
        for e in range(epoch):
            #epoch_loss=0
            ## total numbe rof samples / batch size : tells how many iterations we need basically
            i=0
            while i< len(train_x):
                #print(i)
                start=i
                end=i+batch_size
                batch_x=np.array(train_x[start:end])
                batch_y=np.array(train_y[start:end])
                _,c=sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})
                epoch_loss+=c
                i+=batch_size
            print('Epoch',e, 'completed out of', epoch, 'loss:',epoch_loss)
        ## tf.argmax returns index of maximum values in these arrays
        ## gives us whether prediction and y are identical
        correct=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        #print(correct)
        accuracy=tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy',accuracy.eval({x:test_x,y:test_y}))
         # Save model weights to disk
        saver.save(sess, model_path)
        #print(save_path)
        #print("Model saved in file: %s" % save_path)
        
        
        
        
        
train_neural_network(x)
                





# =============================================================================
# 
# #### WHY DO WE USE SOFTMAX?
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






#with tf.Session() as sess:
#        new_saver = tf.train.import_meta_graph('model.ckpt.meta')
#        
#        sess.run(tf.initialize_all_variables())
#        
#        new_saver.restore(sess, tf.train.latest_checkpoint('./'))
#
#
#        print(sess.run(w1))































