# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:10:44 2019

@author: krish
"""
"""
Function for computing the time
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