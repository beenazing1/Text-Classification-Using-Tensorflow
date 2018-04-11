# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 15:34:30 2018

@author: bsingh46
"""
from TwitterSearch import *
from datetime import *
fout='/Neighborhood_attributes_Classification/safety.txt'    
outF = open(fout, "w", encoding="utf-8")



try:
    tso = TwitterSearchOrder() 
    tso.set_keywords(['safety']) 
    tso.set_language('en') 
    tso.set_include_entities(False) 
   
    tso.set_geocode(33.753746, -84.386330, 2000, imperial_metric=False)

   
    ts = TwitterSearch(
        consumer_key = 'XXX',
        consumer_secret = 'XXX',
        access_token = 'XXX',
        access_token_secret = 'XXX'
     )

   
    for tweet in ts.search_tweets_iterable(tso):
        outF.write(tweet['text'] )
        outF.write('\n')

except TwitterSearchException as e: 
    print(e)
    

      