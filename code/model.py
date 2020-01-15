# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:16:57 2019

@author: krish
"""

from custom_metric import Rsqured

from keras.models import Sequential
from keras.layers import Dense

from keras.regularizers import l2
from keras.layers import Dropout
 
def deep_net(input_shape=(1280)):

    model = Sequential()
    model.add(Dense(4000, activation='relu', input_dim=input_shape, kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.25))
        
    model.add(Dense(2000, activation='relu', input_dim=input_shape, kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.25))
        
    model.add(Dense(1000, activation='relu', input_dim=input_shape, kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.25))

    model.add(Dense(1000, activation='relu', input_dim=input_shape, kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.10))

    model.add(Dense(1, activation=None, use_bias=True, kernel_regularizer=l2(0.0001)))
    
    
    model.compile(optimizer='rmsprop',loss='mean_squared_error', metrics=[Rsqured,'mse'])
    model.summary()
    
    return model
