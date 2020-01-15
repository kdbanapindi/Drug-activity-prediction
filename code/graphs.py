# -*- coding: utf-8 -*-
"""
Created on Mon May  6 11:33:29 2019

@author: krish
"""
import pandas as pd
import numpy as np
import math

import os

from numpy import cumsum

os.chdir('C:/Users/krish/Desktop/STAT 578/data/history')

#history object before feature reduction
data=pd.read_csv('OX1_history.csv')
data=data.drop(['Unnamed: 0'],axis=1)

time=list(data['val_loss'][51:101,])
data=data.loc[0:49,]

data['time']=pd.DataFrame(time)
data['cum_time']=cumsum(data['time'])

#history object after feature reduction
data_red=pd.read_csv('OX1_history_red.csv')

data_red=data_red.drop(['Unnamed: 0'],axis=1)

time=list(data_red['val_loss'][51:101,])
data_red=data_red.iloc[0:49,]

data_red['time']=pd.DataFrame(time)

data_red['cum_time']=cumsum(data_red['time'])



#plotting the graphs

#plot for Rsquared
import matplotlib.pyplot as plt
plt.plot(data['Rsqured'],'r')
plt.plot(data_red['Rsqured'],'b')
plt.title('R_Squared METAB')
plt.xlabel('Number of Iterations ')
plt.ylabel('R^2')

plt.show


#plot for cumulative time
plt.plot(data['cum_time'],'r')
plt.plot(data_red['cum_time'],'b')
plt.title('Cumulative time METAB')
plt.xlabel('Number of Iterations ')
plt.ylabel('Time(sec)')

plt.show

#plot for the loss function
plt.plot(np.log2(data['loss']),'r')
plt.plot(np.log2(data_red['loss']),'b')
plt.title('Loss function of OX1')
plt.xlabel('Number of Iterations ')
plt.ylabel('log(MSE)')

plt.show

