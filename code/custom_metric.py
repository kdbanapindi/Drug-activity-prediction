# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:37:17 2019

@author: krish
"""

"""
Part 2-Defining a custom loss function
"""
import keras.backend as K
def Rsqured(y_true,y_pred):
    

    y_true = K.batch_flatten(y_true)
    y_pred = K.batch_flatten(y_pred)

    y_tr_mean = K.mean(y_true)
    y_pr_mean = K.mean(y_pred)

    num = K.sum((y_true-y_tr_mean) * (y_pred-y_pr_mean))
    num = num*num

    denom = K.sum((y_true-y_tr_mean)*(y_true-y_tr_mean)) * K.sum((y_pred-y_pr_mean)*(y_pred-y_pr_mean))

    return num/denom

