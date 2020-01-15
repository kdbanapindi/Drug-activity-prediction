# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:57:32 2019

@author: krish
"""

save_root='C:/Users/krish/Desktop/STAT 578/data/processed/'
model_root='C:/Users/krish/Desktop/STAT 578/data/models/'

dataset_names = ['DPP4','HIVINT', 'HIVPROT','METAB','OX1']
import json
import os
import pandas as pd

os.chdir('C:/Users/krish/OneDrive/Krishna/Coursework/STAT578/final_project/code')
"""
Class for computing the time
"""
import time
import keras

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

"""
Importing the deep learning model
"""

from model import deep_net

"""
Importing the input features
"""
from inp_feat import feat, red_feat

for i in range(0,len(dataset_names)):


    red_inputs=red_feat(i)
    inputs=feat(i)

    red_model=  deep_net(red_inputs[2])
    model=       deep_net(inputs[2])
    

    time_call = TimeHistory()
    time_call_red=TimeHistory()

    hist_no_red=model.fit(inputs[0], inputs[1], epochs=50, batch_size=300,  verbose=1, validation_split=0.2,callbacks=[time_call])
    
    
    
    hist_red=red_model.fit(red_inputs[0], red_inputs[1], epochs=75, batch_size=100,  verbose=1, validation_split=0.2,callbacks=[time_call_red])
    
    
    os.chdir(model_root)
    
    
    with open(dataset_names[i]+'_history.csv', mode='w') as f:
        pd.DataFrame(hist_no_red.history).to_csv(f)
        pd.DataFrame(time_call.times).to_csv(f)
    

    with open(dataset_names[i]+'_history_red.csv', mode='w') as f:
        pd.DataFrame(hist_red.history).to_csv(f)
        pd.DataFrame(time_call_red.times).to_csv(f)

        
        
    
#plots of all the models
        
import matplotlib.pyplot as plt

history_in = open(dataset_names[i]+'_history_red.json',"rb")



plt.plot(history_old.history['Rsqured'],'r')
plt.plot(history.history['Rsqured'],'b')
plt.title('Evaluation Metrics [R_Squared]')
plt.xlabel('Number of Iterations ')
plt.ylabel('R^2')

plt.show
        
        
