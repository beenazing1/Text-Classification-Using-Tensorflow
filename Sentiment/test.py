# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 12:12:58 2018

@author: bsingh46
"""

import tensorflow as tf
import pickle
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from  tf_text import create_feature_set_and_labels

train_x,train_y,test_x, test_y=create_feature_set_and_labels('negative.txt','positive.txt')
model_path = "model/model.ckpt"

n_h1=5000
n_h2=5000
n_h3=5000

n_class=2
batch_size=25
tf.set_random_seed(1234)
x=tf.placeholder('float',[None,len(train_x[0])])
print(len(train_x[0]))
y=tf.placeholder('float')
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
    
    tf.set_random_seed(1234)

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

#x=tf.placeholder('float',[None,len(train_x[0])])


saver=tf.train.Saver()


def use_neural_network(input_data):
    prediction = neural_network_model(x)
    print(prediction)
    with open('lexicon.pickle','rb') as f:
        lexicon = pickle.load(f)
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,"model/model.ckpt")
        current_words = word_tokenize(input_data.lower())
        current_words = [lemmatizer.lemmatize(i) for i in current_words]
        features = np.zeros(len(lexicon))

        for word in current_words:
            if word.lower() in lexicon:
                index_value = lexicon.index(word.lower())
                # OR DO +=1, test both
                features[index_value] += 1

        features = np.array(list(features))
        # pos: [1,0] , argmax: 0
        # neg: [0,1] , argmax: 1
        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}),1)))
        print(prediction.eval(feed_dict={x:[features]}))
        if result[0] == 1:
            print('Positive:',input_data)
        elif result[0] == 0:
            print('Negative:',input_data)

use_neural_network("STUCK IN TRAFFIC DUE TO MAJOR ACCIDENT.")
use_neural_network("This was the best weather i've ever seen.")
