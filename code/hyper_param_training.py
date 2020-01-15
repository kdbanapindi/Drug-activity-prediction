# -*- coding: utf-8 -*-
"""
Created on Wed May  1 18:44:56 2019

@author: krish
"""

import numpy as np
from keras import models
from keras import layers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from custom_metric import Rsqured

from keras.models import Sequential
from keras.layers import Dense

from keras.regularizers import l2
from keras.layers import Dropout

import json
import pandas as pd
import numpy as np

save_root='C:/Users/krish/Desktop/STAT 578/data/processed/'
dataset_names = ['DPP4','HIVINT', 'HIVPROT','METAB','OX1']
best_ind_root='C:/Users/krish/Desktop/STAT 578/data/Best_indices/'


inp_data=pd.read_csv(save_root+dataset_names[i]+'_train_processed.csv')
inp_data.set_index('MOLECULE', inplace=True)
best_ind=pd.read_csv(best_ind_root+dataset_names[i]+'_best_ind.csv').values.transpose()
ind=np.array(best_ind, dtype=bool)
    
X=inp_data.loc[:, inp_data.columns != 'Act'].values
features=X[:,ind[0,:]]
y=inp_data.loc[:,inp_data.columns=='Act'].values
target=np.reshape(y,(-1,1))
input_shape=features.shape[1]





def create_network(optimizer='rmsprop'):
    
    # Start neural network
    model=Sequential()

    # Add fully connected layer with a ReLU activation function
    model.add(Dense(units=4000, activation='relu', input_shape=(input_shape,), kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.25))
    # Add fully connected layer with a ReLU activation function
    model.add(Dense(units=2000, activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.25))
    
    
    model.add(Dense(units=1000, activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.25))
    
    model.add(Dense(units=1000, activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.1))
    # Add fully connected layer with a sigmoid activation function
    model.add(Dense(units=1, activation=None,use_bias=True, kernel_regularizer=l2(0.0001)))

    # Compile neural network
    model.compile(loss='mean_squared_error', # Cross-entropy
                    optimizer=optimizer, # Optimizer
                    metrics=['mse',Rsqured]) # Accuracy performance metric
    
    # Return compiled network
    return model


neural_network = KerasRegressor(build_fn=create_network, verbose=1)


epochs = [75]
batches = [100,200,300]
optimizers = ['rmsprop','Adam']

# Create hyperparameter options
hyperparameters = dict(optimizer=optimizers, epochs=epochs, batch_size=batches)

# Create grid search
grid = GridSearchCV(estimator=neural_network, param_grid=hyperparameters)

# Fit grid search
grid_result = grid.fit(features, target,validation_split=0.2)

 
print("Best: %r using %s" % (grid_result.best_estimator_, grid_result.best_params_))




from sklearn.externals import joblib

joblib.dump(grid_result, 'grid_results.pkl')


with open('grid_search.json', 'w') as f:
        json.dump(grid_result, f)
   
import pickle     
        
pickle_in = open("grid_results.pkl","rb")
example_grid = pickle.load(pickle_in)
        
