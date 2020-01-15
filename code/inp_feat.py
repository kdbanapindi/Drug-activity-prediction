# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:25:33 2019

@author: krish
"""

import pandas as pd
import numpy as np

save_root='C:/Users/krish/Desktop/STAT 578/data/processed/'
dataset_names = ['DPP4','HIVINT', 'HIVPROT','METAB','OX1']
best_ind_root='C:/Users/krish/Desktop/STAT 578/data/Best_indices/'


def feat(i):
    inp_data=pd.read_csv(save_root+dataset_names[i]+'_train_processed.csv')
    inp_data.set_index('MOLECULE', inplace=True)
    X=inp_data.loc[:, inp_data.columns != 'Act'].values
    y=inp_data.loc[:,inp_data.columns=='Act'].values
    y=np.reshape(y,(-1,1))
    input_shape=X.shape[1]
    return(X,y,input_shape)
    
def red_feat(i):
    
    inp_data=pd.read_csv(save_root+dataset_names[i]+'_train_processed.csv')
    inp_data.set_index('MOLECULE', inplace=True)
    best_ind=pd.read_csv(best_ind_root+dataset_names[i]+'_best_ind.csv').values.transpose()
    ind=np.array(best_ind, dtype=bool)
    
    X=inp_data.loc[:, inp_data.columns != 'Act'].values
    X=X[:,ind[0,:]]
    y=inp_data.loc[:,inp_data.columns=='Act'].values
    y=np.reshape(y,(-1,1))
    input_shape=X.shape[1]
    return(X,y,input_shape)