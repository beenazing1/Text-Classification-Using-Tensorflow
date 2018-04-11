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
    tso = TwitterSearchOrder() # create a TwitterSearchOrder object
    tso.set_keywords(['safety']) # let's define all words we would like to have a look for
    tso.set_language('en') # we want to see German tweets only
    tso.set_include_entities(False) # and don't give us all those entity information
    #tso.set_until(date(2017, 12, 24))

    #tso.set_negative_attitude_filter()
    tso.set_geocode(33.753746, -84.386330, 2000, imperial_metric=False)

    # it's about time to create a TwitterSearch object with our secret tokens
    ts = TwitterSearch(
        consumer_key = 'XXX',
        consumer_secret = 'XXX',
        access_token = 'XXX',
        access_token_secret = 'XXX'
     )

     # this is where the fun actually starts :)
    for tweet in ts.search_tweets_iterable(tso):
        outF.write(tweet['text'] )
        outF.write('\n')

except TwitterSearchException as e: # take care of all those ugly errors if there are some
    print(e)
    

      